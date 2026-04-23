from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftEvent:
    step: int
    drift_type: str
    score: float
    threshold: float
    severity: float
    detector: str


class BaseDriftDetector(ABC):
    @abstractmethod
    def update(self, *args, **kwargs) -> Optional[DriftEvent]:
        raise NotImplementedError
