from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

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
