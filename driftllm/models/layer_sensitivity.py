from collections import defaultdict
from typing import Dict, List

import torch


class LayerSensitivityAnalyzer:
    DRIFT_TYPE_MODULES = {
        "semantic_drift": ["q_proj", "k_proj", "v_proj"],
        "label_drift": ["score", "classifier", "down_proj"],
        "knowledge_drift": ["gate_proj", "up_proj", "down_proj"],
    }

    def __init__(self, top_k: int = 8, min_score: float = 0.0):
        self.top_k = top_k
        self.min_score = min_score

    def compute_fisher(self, model, dataloader, n_samples: int = 128) -> Dict[str, float]:
        model.train()
        device = next(model.parameters()).device
        scores = defaultdict(float)
        n_batches = 0
        for batch in dataloader:
            if n_batches * len(batch["labels"]) >= n_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    scores[n] += p.grad.detach().pow(2).mean().item()
            model.zero_grad(set_to_none=True)
            n_batches += 1
        return {k: v / max(1, n_batches) for k, v in scores.items()}

    def select_layers(self, fisher: Dict[str, float], drift_type: str) -> List[str]:
        keys = self.DRIFT_TYPE_MODULES[drift_type]
        filt = {k: v for k, v in fisher.items() if any(x in k for x in keys)}
        ranked = sorted(filt.items(), key=lambda x: x[1], reverse=True)
        selected = [k for k, v in ranked if v > self.min_score][: self.top_k]
        if not selected:
            fallback = sorted(fisher.items(), key=lambda x: x[1], reverse=True)
            selected = [k for k, v in fallback if v > self.min_score][: self.top_k]
        print(f"[LayerSensitivity] drift_type={drift_type} selected={[(k, fisher[k]) for k in selected]}")
        return selected
