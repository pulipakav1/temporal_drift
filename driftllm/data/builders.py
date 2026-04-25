from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from driftllm.data.dataset import (
    load_agnews_dataset,
    load_amazon_dataset,
    load_arxiv_dataset,
    load_financial_dataset,
    load_mimic_dataset,
    load_tweeteval_dataset,
)


class TokenizedDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "date": row.get("date", ""),
            "drift_event": row.get("drift_event", "none"),
        }


def _collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "date": [x["date"] for x in batch],
        "drift_event": [x["drift_event"] for x in batch],
    }


def build_domain_splits(cfg) -> Dict[str, object]:
    domain = cfg["experiment"]["domain"]
    if domain == "financial":
        return load_financial_dataset(cfg["paths"]["financial_cache"], seed=int(cfg["experiment"]["seed"]))
    if domain == "tweeteval":
        return load_tweeteval_dataset(cfg["paths"]["tweet_cache"], seed=int(cfg["experiment"]["seed"]))
    if domain == "agnews":
        return load_agnews_dataset(cfg["paths"].get("agnews_cache", "./artifacts/data/agnews_hf"), seed=int(cfg["experiment"]["seed"]))
    if domain == "amazon":
        return load_amazon_dataset(cfg["paths"].get("amazon_cache", "./artifacts/data/amazon_hf"), seed=int(cfg["experiment"]["seed"]))
    if domain == "arxiv":
        return load_arxiv_dataset(cfg["paths"]["arxiv_cache"], seed=int(cfg["experiment"]["seed"]))
    return load_mimic_dataset(cfg["paths"]["mimic_path"])


def build_stream_loader(cfg, tokenizer, split: str = "test", batch_size: int = 1) -> DataLoader:
    splits = build_domain_splits(cfg)
    if split == "validation_test":
        rows = [dict(r) for r in splits["validation"]] + [dict(r) for r in splits["test"]]
    elif split == "full_stream":
        rows = [dict(r) for r in splits["train"]] + [dict(r) for r in splits["validation"]] + [dict(r) for r in splits["test"]]
    else:
        rows = [dict(r) for r in splits[split]]
    max_length = 256 if cfg["experiment"]["domain"] == "amazon" else 256
    ds = TokenizedDataset(rows, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)


def build_adaptation_loader(buffer_samples: List[Dict[str, torch.Tensor]], batch_size: int = 8) -> Optional[DataLoader]:
    if not buffer_samples:
        return None

    class _Tensors(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    def _collate_tensor_dict(batch):
        return {k: torch.stack([x[k] for x in batch], dim=0) for k in batch[0].keys()}

    return DataLoader(_Tensors(buffer_samples), batch_size=batch_size, shuffle=True, collate_fn=_collate_tensor_dict)
