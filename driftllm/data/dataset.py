import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from driftllm.data.drift_simulator import inject_financial_drift
from driftllm.data.temporal_splitter import temporal_split


def _pseudo_dates(n: int):
    start, end = date(2020, 1, 1), date(2024, 12, 31)
    span = (end - start).days
    return [(start + timedelta(days=int(i * span / max(1, n - 1)))).isoformat() for i in range(n)]


def load_financial_dataset(cache_dir: str, seed: int = 42) -> DatasetDict:
    cache = Path(cache_dir)
    if cache.exists():
        return load_from_disk(str(cache))
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)["train"]
    ds = ds.rename_column("sentence", "text")
    ds = ds.add_column("date", _pseudo_dates(len(ds)))
    ds = ds.add_column("source", ["financial_phrasebank"] * len(ds))
    ds = ds.add_column("drift_event", ["none"] * len(ds))
    ds = ds.add_column("drift_provenance", ["none"] * len(ds))
    ds = ds.map(lambda x: inject_financial_drift(x, seed=seed))
    out = temporal_split(ds)
    out.save_to_disk(str(cache))
    return out


def load_mimic_dataset(path: str) -> DatasetDict:
    df = pd.read_csv(path)
    df = df.rename(columns={"note": "text", "timestamp": "date", "code": "label"})
    if "drift_event" not in df.columns:
        df["drift_event"] = "none"
    if "drift_provenance" not in df.columns:
        df["drift_provenance"] = "clinical_source_or_annotation"
    df["source"] = "mimic_iii"
    ds = Dataset.from_pandas(df[["text", "label", "date", "drift_event", "drift_provenance", "source"]], preserve_index=False)
    return temporal_split(ds)


_ARXIV_DATE_RE = re.compile(r"\]\s+(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})")


def _extract_arxiv_date(text: str) -> str | None:
    match = _ARXIV_DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d %b %Y").date().isoformat()
    except ValueError:
        return None


def _annotate_arxiv_drift(iso_date: str) -> tuple[str, str]:
    x = datetime.fromisoformat(iso_date).date()
    if date(2022, 2, 1) <= x <= date(2022, 5, 31):
        return "representation_shift_2022", "older_arxiv_topics"
    if date(2023, 3, 1) <= x <= date(2023, 6, 30):
        return "deep_learning_surge_2023", "neural_methods_growth"
    if date(2024, 2, 1) <= x <= date(2024, 5, 31):
        return "systems_scaling_2024", "large_scale_systems_and_optimization"
    return "none", "none"


def load_arxiv_dataset(cache_dir: str, seed: int = 42) -> DatasetDict:
    del seed  # arXiv loading is deterministic; the signature stays aligned with other loaders.
    cache = Path(cache_dir)
    if cache.exists():
        return load_from_disk(str(cache))

    ds = load_dataset("ccdv/arxiv-classification", "no_ref")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    label_names = combined.features["label"].names

    observed_dates = [_extract_arxiv_date(str(row["text"])) for row in combined]
    fallback_dates = _pseudo_dates(len(combined))
    if all(d is not None for d in observed_dates):
        parsed = [datetime.fromisoformat(d).date() for d in observed_dates if d is not None]
        src_min, src_max = min(parsed), max(parsed)
        src_span = max(1, (src_max - src_min).days)
        dst_min, dst_max = date(2020, 1, 1), date(2024, 12, 31)
        dst_span = (dst_max - dst_min).days

        def _project_date(iso_date: str) -> str:
            current = datetime.fromisoformat(iso_date).date()
            rel = (current - src_min).days / src_span
            return (dst_min + timedelta(days=int(rel * dst_span))).isoformat()

    else:
        def _project_date(iso_date: str) -> str:
            return iso_date

    rows = []
    for idx, row in enumerate(combined):
        text = str(row["text"])
        raw_date = _extract_arxiv_date(text) or fallback_dates[idx]
        iso_date = _project_date(raw_date)
        drift_event, provenance = _annotate_arxiv_drift(iso_date)
        rows.append(
            {
                "text": text,
                "label": int(row["label"]),
                "label_name": label_names[int(row["label"])],
                "date": iso_date,
                "drift_event": drift_event,
                "drift_provenance": provenance,
                "source": "ccdv/arxiv-classification",
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = temporal_split(Dataset.from_pandas(frame, preserve_index=False))
    out.save_to_disk(str(cache))
    return out
