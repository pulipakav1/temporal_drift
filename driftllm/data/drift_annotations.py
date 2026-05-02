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

AGNEWS_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("covid_world_surge_2020", date(2020, 3, 1), date(2020, 5, 31), "label_drift", "agnews"),
    DriftAnnotation("us_election_world_2020", date(2020, 10, 15), date(2020, 11, 30), "label_drift", "agnews"),
    DriftAnnotation("ukraine_world_surge_2022", date(2022, 2, 24), date(2022, 4, 30), "label_drift", "agnews"),
    DriftAnnotation("ai_scitech_boom_2022", date(2022, 11, 1), date(2023, 3, 31), "semantic_drift", "agnews"),
]

AMAZON_EVENTS: List[DriftAnnotation] = [
    DriftAnnotation("covid_shopping_shift_2020", date(2020, 3, 1), date(2020, 5, 31), "semantic_drift", "amazon"),
    DriftAnnotation("supply_chain_crisis_2021", date(2021, 8, 1), date(2021, 12, 31), "label_drift", "amazon"),
    DriftAnnotation("post_covid_return_2022", date(2022, 3, 1), date(2022, 6, 30), "semantic_drift", "amazon"),
    DriftAnnotation("ai_review_shift_2023", date(2023, 1, 1), date(2023, 6, 30), "knowledge_drift", "amazon"),
]


def event_name_to_type(domain: str) -> Dict[str, str]:
    if domain == "financial":
        events = FINANCIAL_EVENTS
    elif domain == "tweeteval":
        events = TWEET_EVENTS
    elif domain == "agnews":
        events = AGNEWS_EVENTS
    elif domain == "amazon":
        events = AMAZON_EVENTS
    elif domain == "arxiv":
        events = ARXIV_EVENTS
    else:
        events = []
    return {e.name: e.drift_type for e in events}
