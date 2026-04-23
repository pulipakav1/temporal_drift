import random
from collections import deque
from copy import deepcopy

import torch


class ForgettingRegularizer:
    def __init__(self, ewc_lambda: float = 0.5, replay_buffer_size: int = 512):
        self.ewc_lambda = ewc_lambda
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.fisher_diag = {}
        self.optimal_params = {}

    def consolidate(self, model, dataloader, device, n_samples: int = 256):
        model.train()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        seen = 0
        for batch in dataloader:
            if seen >= n_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            model.zero_grad(set_to_none=True)
            seen += int(batch["labels"].shape[0])
        self.fisher_diag = {n: f / max(1, seen) for n, f in fisher.items()}
        self.optimal_params = {n: deepcopy(p.detach()) for n, p in model.named_parameters() if n in self.fisher_diag}
        print("[EWC] consolidation complete")

    def ewc_loss(self, model):
        if not self.fisher_diag:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher_diag:
                loss = loss + (self.fisher_diag[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * 0.5 * loss

    def add_to_replay(self, sample):
        self.replay_buffer.append(sample)

    def get_replay_batch(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return None
        xs = random.sample(self.replay_buffer, batch_size)
        out = {}
        for k in xs[0]:
            out[k] = torch.stack([x[k] for x in xs], dim=0)
        return out
