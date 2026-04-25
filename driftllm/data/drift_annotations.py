from dataclasses import dataclass
from datetime import date
from typing import Dict, List


@dataclass
class DriftAnnotation:
    name: str
    start: date
    end: date
    drift_type: str
    domain: str


FINANCIAL_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("covid_crash_mar_2020", date(2020, 3, 1), date(2020, 5, 31), "label_drift", "financial"),
    DriftAnnotation("fed_rate_hike_mar_2022", date(2022, 3, 1), date(2022, 4, 30), "label_drift", "financial"),
    DriftAnnotation("crypto_winter_nov_2022", date(2022, 11, 1), date(2022, 12, 31), "semantic_drift", "financial"),
    DriftAnnotation("svb_collapse_mar_2023", date(2023, 3, 1), date(2023, 4, 15), "knowledge_drift", "financial"),
    DriftAnnotation("fed_pivot_sep_2024", date(2024, 9, 1), date(2024, 10, 15), "knowledge_drift", "financial"),
]

CLINICAL_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("icd10_transition_2021", date(2021, 1, 1), date(2021, 3, 31), "label_drift", "clinical"),
    DriftAnnotation("covid_treatment_shift_2021", date(2021, 8, 1), date(2021, 10, 31), "knowledge_drift", "clinical"),
]

ARXIV_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("representation_shift_2022", date(2022, 2, 1), date(2022, 5, 31), "semantic_drift", "arxiv"),
    DriftAnnotation("deep_learning_surge_2023", date(2023, 3, 1), date(2023, 6, 30), "label_drift", "arxiv"),
    DriftAnnotation("systems_scaling_2024", date(2024, 2, 1), date(2024, 5, 31), "knowledge_drift", "arxiv"),
]

TWEET_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("covid_language_shift_2020", date(2020, 3, 1), date(2020, 6, 30), "semantic_drift", "tweeteval"),
    DriftAnnotation("us_election_sentiment_2020", date(2020, 10, 1), date(2020, 12, 15), "label_drift", "tweeteval"),
    DriftAnnotation("ukraine_news_cycle_2022", date(2022, 2, 1), date(2022, 4, 30), "semantic_drift", "tweeteval"),
    DriftAnnotation("llm_discourse_shift_2022", date(2022, 11, 1), date(2023, 2, 28), "knowledge_drift", "tweeteval"),
]


def event_name_to_type(domain: str) -> Dict[str, str]:
    if domain == "financial":
        events = FINANCIAL_EVENTS
    elif domain == "tweeteval":
        events = TWEET_EVENTS
    elif domain == "arxiv":
        events = ARXIV_EVENTS
    else:
        events = CLINICAL_EVENTS
    return {e.name: e.drift_type for e in events}
