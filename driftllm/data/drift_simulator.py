from datetime import datetime
import hashlib

import random


def inject_financial_drift(example, seed: int = 42):
    stable_offset = int(hashlib.md5(str(example["date"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_offset)
    d = datetime.fromisoformat(example["date"]).date()
    label = example["label"]
    event = "none"
    provenance = "none"
    if datetime(2020, 3, 1).date() <= d <= datetime(2020, 5, 31).date():
        event = "covid_crash_mar_2020"
        # Increase negative sentiment by ~40% (probabilistic relabeling).
        if label != 0 and rng.random() < 0.4:
            label = 0
            provenance = "covid_neg_boost_40pct"
    if datetime(2022, 3, 1).date() <= d <= datetime(2022, 4, 30).date():
        event = "fed_rate_hike_mar_2022"
        if label == 1 and rng.random() < 0.5:
            label = 0
            provenance = "fed_neutral_to_negative_shift"
    return {"label": label, "drift_event": event, "drift_provenance": provenance}
