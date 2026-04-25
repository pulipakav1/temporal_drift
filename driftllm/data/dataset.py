import re
import hashlib
import random
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
    try:
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")["train"]
        ds = ds.rename_column("sentence", "text")
        source_name = "financial_phrasebank"
    except Exception:
        # Fallback for environments where legacy dataset scripts are blocked.
        alt = load_dataset("zeroshot/twitter-financial-news-sentiment")
        ds = concatenate_datasets([alt["train"], alt["validation"]])
        source_name = "twitter_financial_news_sentiment"
    ds = ds.add_column("date", _pseudo_dates(len(ds)))
    ds = ds.add_column("source", [source_name] * len(ds))
    ds = ds.add_column("drift_event", ["none"] * len(ds))
    ds = ds.add_column("drift_provenance", ["none"] * len(ds))
    ds = ds.map(lambda x: inject_financial_drift(x, seed=seed))
    out = temporal_split(ds)
    out.save_to_disk(str(cache))
    return out


def _annotate_tweet_drift(iso_date: str) -> tuple[str, str]:
    x = datetime.fromisoformat(iso_date).date()
    if date(2020, 3, 1) <= x <= date(2020, 6, 30):
        return "covid_language_shift_2020", "pandemic_discourse_shift"
    if date(2020, 10, 1) <= x <= date(2020, 12, 15):
        return "us_election_sentiment_2020", "political_polarization_shift"
    if date(2022, 2, 1) <= x <= date(2022, 4, 30):
        return "ukraine_news_cycle_2022", "geopolitical_topic_shift"
    if date(2022, 11, 1) <= x <= date(2023, 2, 28):
        return "llm_discourse_shift_2022", "generative_ai_topic_shift"
    return "none", "none"


def load_tweeteval_dataset(cache_dir: str, seed: int = 42) -> DatasetDict:
    del seed  # deterministic data loading
    cache = Path(cache_dir)
    if cache.exists():
        return load_from_disk(str(cache))

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    pseudo_dates = _pseudo_dates(len(combined))
    rows = []
    for idx, row in enumerate(combined):
        iso_date = pseudo_dates[idx]
        drift_event, provenance = _annotate_tweet_drift(iso_date)
        rows.append(
            {
                "text": str(row["text"]),
                "label": int(row["label"]),
                "date": iso_date,
                "drift_event": drift_event,
                "drift_provenance": provenance,
                "source": "cardiffnlp/tweet_eval_sentiment",
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = temporal_split(Dataset.from_pandas(frame, preserve_index=False))
    out.save_to_disk(str(cache))
    return out


def _annotate_agnews_drift(iso_date: str) -> tuple[str, str]:
    x = datetime.fromisoformat(iso_date).date()
    if date(2020, 3, 1) <= x <= date(2020, 5, 31):
        return "covid_world_surge_2020", "covid_world_label_boost_40pct"
    if date(2020, 10, 15) <= x <= date(2020, 11, 30):
        return "us_election_world_2020", "election_world_label_boost_30pct"
    if date(2022, 2, 24) <= x <= date(2022, 4, 30):
        return "ukraine_world_surge_2022", "ukraine_world_label_boost_35pct"
    if date(2022, 11, 1) <= x <= date(2023, 3, 31):
        return "ai_scitech_boom_2022", "ai_scitech_label_boost_30pct"
    return "none", "none"


def _inject_agnews_drift(example, seed: int = 42):
    stable_offset = int(hashlib.md5(str(example["date"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_offset)
    d = datetime.fromisoformat(example["date"]).date()
    label = int(example["label"])
    event = "none"
    provenance = "none"
    if date(2020, 3, 1) <= d <= date(2020, 5, 31):
        event = "covid_world_surge_2020"
        if label in {1, 2} and rng.random() < 0.40:
            label = 0
            provenance = "covid_world_label_boost_40pct"
    if date(2020, 10, 15) <= d <= date(2020, 11, 30):
        event = "us_election_world_2020"
        if label in {1, 2, 3} and rng.random() < 0.30:
            label = 0
            provenance = "election_world_label_boost_30pct"
    if date(2022, 2, 24) <= d <= date(2022, 4, 30):
        event = "ukraine_world_surge_2022"
        if label in {1, 2, 3} and rng.random() < 0.35:
            label = 0
            provenance = "ukraine_world_label_boost_35pct"
    if date(2022, 11, 1) <= d <= date(2023, 3, 31):
        event = "ai_scitech_boom_2022"
        if label == 2 and rng.random() < 0.30:
            label = 3
            provenance = "ai_scitech_label_boost_30pct"
    return {"label": label, "drift_event": event, "drift_provenance": provenance}


def load_agnews_dataset(cache_dir: str, seed: int = 42) -> DatasetDict:
    cache = Path(cache_dir)
    if cache.exists():
        return load_from_disk(str(cache))
    ds = load_dataset("ag_news")
    combined = concatenate_datasets([ds["train"], ds["test"]])
    combined = combined.add_column("date", _pseudo_dates(len(combined)))
    combined = combined.add_column("source", ["ag_news"] * len(combined))
    combined = combined.map(lambda x: _inject_agnews_drift(x, seed=seed))
    out = temporal_split(combined)
    out.save_to_disk(str(cache))
    return out


def _timestamp_ms_to_iso(ms_value) -> str | None:
    try:
        value = int(ms_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return datetime.utcfromtimestamp(value / 1000.0).date().isoformat()


def _rating_to_sentiment_label(rating: int) -> int:
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def _annotate_amazon_drift(iso_date: str) -> tuple[str, str]:
    x = datetime.fromisoformat(iso_date).date()
    if date(2020, 3, 1) <= x <= date(2020, 5, 31):
        return (
            "covid_shopping_shift_2020",
            "COVID lockdown shifts Amazon purchasing patterns and review language",
        )
    if date(2021, 8, 1) <= x <= date(2021, 12, 31):
        return (
            "supply_chain_crisis_2021",
            "Supply chain disruptions — negative reviews surge for delayed items",
        )
    if date(2022, 3, 1) <= x <= date(2022, 6, 30):
        return (
            "post_covid_return_2022",
            "Return to normal purchasing — review language and topics shift",
        )
    if date(2023, 1, 1) <= x <= date(2023, 6, 30):
        return (
            "ai_review_shift_2023",
            "AI/tech product surge — new product category terminology",
        )
    return "none", "none"


def _inject_amazon_drift(example, seed: int = 42):
    stable_offset = int(hashlib.md5(str(example["date"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_offset)
    label = int(example["label"])
    event, provenance = _annotate_amazon_drift(example["date"])
    if event == "supply_chain_crisis_2021" and label == 1 and rng.random() < 0.25:
        label = 0
        provenance = "supply_chain_neutral_to_negative_25pct"
    return {"label": label, "drift_event": event, "drift_provenance": provenance}


def load_amazon_dataset(cache_dir: str, seed: int = 42, max_records: int = 50000) -> DatasetDict:
    cache = Path(cache_dir)
    if cache.exists():
        return load_from_disk(str(cache))

    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", trust_remote_code=True)["full"]
    rows = []
    for row in ds:
        iso_date = _timestamp_ms_to_iso(row.get("timestamp"))
        if iso_date is None:
            continue
        d = datetime.fromisoformat(iso_date).date()
        if not (date(2020, 1, 1) <= d <= date(2024, 12, 31)):
            continue
        rows.append(
            {
                "text": str(row.get("text", "")),
                "label": _rating_to_sentiment_label(int(row.get("rating", 3))),
                "date": iso_date,
                "source": "amazon_reviews_2023_books",
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    if len(frame) > max_records:
        counts = frame["year"].value_counts().sort_index()
        total = int(counts.sum())
        targets = {int(y): int(max_records * int(c) / total) for y, c in counts.items()}
        remaining = int(max_records - sum(targets.values()))
        remainders = sorted(
            ((int(y), (max_records * int(c) / total) - targets[int(y)]) for y, c in counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        for y, _ in remainders[:remaining]:
            targets[y] += 1
        sampled = []
        for y, target in targets.items():
            part = frame[frame["year"] == y]
            sampled.append(part.sample(n=min(target, len(part)), random_state=seed))
        frame = pd.concat(sampled, axis=0).sort_values("date").reset_index(drop=True)
    frame = frame.drop(columns=["year"])
    enriched = Dataset.from_pandas(frame, preserve_index=False).map(lambda x: _inject_amazon_drift(x, seed=seed))
    out = temporal_split(enriched)
    print(
        f"[amazon] split sizes train={len(out['train'])} validation={len(out['validation'])} test={len(out['test'])}"
    )
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
