from datetime import datetime
from typing import Dict

from datasets import Dataset, DatasetDict


def temporal_split(dataset: Dataset) -> DatasetDict:
    def in_range(d: str, left: str, right: str) -> bool:
        x = datetime.fromisoformat(d).date()
        return datetime.fromisoformat(left).date() <= x < datetime.fromisoformat(right).date()

    train = dataset.filter(lambda x: in_range(x["date"], "2020-01-01", "2022-01-01"))
    val = dataset.filter(lambda x: in_range(x["date"], "2022-01-01", "2023-01-01"))
    test = dataset.filter(lambda x: in_range(x["date"], "2023-01-01", "2027-01-01"))
    return DatasetDict(train=train, validation=val, test=test)
