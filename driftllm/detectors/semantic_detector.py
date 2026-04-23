from collections import deque
from typing import Optional

import torch

from driftllm.detectors.base_detector import BaseDriftDetector, DriftEvent


class RandomProjection(torch.nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int = 256):
        super().__init__()
        mat = torch.randn(hidden_dim, embedding_dim) / embedding_dim**0.5
        self.register_buffer("proj", mat)

    def forward(self, x):
        return x @ self.proj


class SemanticDriftDetector(BaseDriftDetector):
    def __init__(self, threshold: float, ref_size: int = 1000, window_size: int = 500):
        self.threshold = threshold
        self.ref = deque(maxlen=ref_size)
        self.window = deque(maxlen=window_size)
        self.projector = None

    @staticmethod
    def _rbf(x, y, sigma):
        d2 = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-d2 / (2 * sigma**2 + 1e-8))

    def _mmd_unbiased(self, x, y):
        d = torch.cdist(torch.cat([x, y]), torch.cat([x, y]), p=2)
        sigma = torch.median(d[d > 0]).clamp_min(1e-6)
        kxx = self._rbf(x, x, sigma)
        kyy = self._rbf(y, y, sigma)
        kxy = self._rbf(x, y, sigma)
        n, m = x.shape[0], y.shape[0]
        return ((kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
                + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                - 2 * kxy.mean()).item()

    def update(self, embedding: torch.Tensor, step: int) -> Optional[DriftEvent]:
        x = embedding.detach().float().cpu().view(-1)
        if self.projector is None or self.projector.proj.shape[0] != x.numel():
            self.projector = RandomProjection(hidden_dim=x.numel())
        z = self.projector(x).view(-1)
        self.window.append(z)
        if len(self.ref) < self.ref.maxlen:
            self.ref.append(z)
            return None
        if step % 50 != 0 or len(self.window) < 200:
            return None
        r = torch.stack(list(self.ref))
        w = torch.stack(list(self.window))
        idx_r = torch.randperm(r.size(0))[:200]
        idx_w = torch.randperm(w.size(0))[:200]
        mmd = self._mmd_unbiased(r[idx_r], w[idx_w])
        if mmd > self.threshold:
            sev = min(1.0, mmd / (self.threshold * 5))
            return DriftEvent(step, "semantic_drift", mmd, self.threshold, sev, "semantic")
        return None

    def update_reference(self):
        self.ref = deque(list(self.window), maxlen=self.ref.maxlen)
