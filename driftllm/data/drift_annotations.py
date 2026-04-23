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


def event_name_to_type(domain: str) -> Dict[str, str]:
    events = FINANCIAL_EVENTS if domain == "financial" else CLINICAL_EVENTS
    return {e.name: e.drift_type for e in events}
