import math
from collections import deque
from typing import List, Optional

import numpy as np

from driftllm.detectors.base_detector import BaseDriftDetector, DriftEvent


class ADWIN:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()

    def add(self, x: float) -> bool:
        self.window.append(float(x))
        n = len(self.window)
        if n < 50:
            return False
        arr = np.array(self.window, dtype=np.float32)
        for cut in range(10, n - 10):
            l, r = arr[:cut], arr[cut:]
            n0, n1 = len(l), len(r)
            m = 1 / (1 / n0 + 1 / n1)
            eps = math.sqrt(math.log(4 * n / self.delta) / (2 * m))
            if abs(l.mean() - r.mean()) > eps:
                self.window = deque(arr[cut:].tolist())
                return True
        return False


class LabelDriftDetector(BaseDriftDetector):
    def __init__(self, n_classes: int = 3, delta: float = 0.002):
        self.n_classes = n_classes
        self.detectors: List[ADWIN] = [ADWIN(delta=delta) for _ in range(n_classes)]
        self.freq = np.zeros(n_classes, dtype=np.float32)
        self.n = 0

    def update(self, pred_label: int, step: int) -> Optional[DriftEvent]:
        self.n += 1
        self.freq[pred_label] += 1
        fired = False
        one_hot = np.zeros(self.n_classes)
        one_hot[pred_label] = 1.0
        for i, d in enumerate(self.detectors):
            fired |= d.add(one_hot[i])
        if not fired:
            return None
        p = self.freq / max(1, self.n)
        uniform = np.ones_like(p) / self.n_classes
        sev = float(min(1.0, np.abs(p - uniform).sum()))
        self.freq *= 0
        self.n = 0
        return DriftEvent(step, "label_drift", sev, 0.0, sev, "label_adwin")
