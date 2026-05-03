from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm

from driftllm.data.builders import build_domain_splits
from driftllm.models.base_model import load_model_tokenizer


class _TokenizedDataset(TorchDataset):
    def __init__(self, rows, tok, max_length: int = 256):
        self.rows = rows
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tok(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(int(row["label"]))


class InitialTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def _num_labels(self) -> int:
        domain = self.cfg["experiment"]["domain"]
        if domain in ("financial", "twitter", "tweeteval"):
            return int(self.cfg["model"]["num_labels_financial"])
        if domain == "agnews":
            return int(self.cfg["model"]["num_labels_agnews"])
        if domain == "amazon":
            return int(self.cfg["model"].get("num_labels_amazon", self.cfg["model"]["num_labels_financial"]))
        if domain == "arxiv":
            return int(self.cfg["model"]["num_labels_arxiv"])
        raise ValueError(f"Unknown domain: {domain}")

    def run(self):
        splits = build_domain_splits(self.cfg)
        n_labels = self._num_labels()
        model, tok = load_model_tokenizer(
            self.cfg["model"]["name"], n_labels, self.cfg["model"]["bf16"], self.cfg["model"]["device_map"]
        )
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(target_device)
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in ("score", "classifier"))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        device = next(model.parameters()).device

        batch_size = int(self.cfg["training"].get("initial_batch_size", 32))
        train_rows = [dict(x) for x in splits["train"]]
        val_rows = [dict(x) for x in splits["validation"]]
        train_loader = DataLoader(
            _TokenizedDataset(train_rows, tok), batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            _TokenizedDataset(val_rows, tok), batch_size=batch_size * 2, shuffle=False, pin_memory=True
        )

        opt = torch.optim.AdamW(trainable_params, lr=self.cfg["training"]["lr"])
        patience = int(self.cfg["training"]["early_stopping_patience"])
        best_val = -1.0
        bad_epochs = 0
        out_dir = Path(self.cfg["paths"]["model_dir"]) / self.cfg["experiment"]["domain"] / "initial"
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(int(self.cfg["training"].get("initial_epochs", 3))):
            model.train()
            train_iter = tqdm(train_loader, desc=f"initial-train epoch {epoch + 1}", leave=False)
            for step, (enc, labels) in enumerate(train_iter, start=1):
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                loss = model(**enc, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, float(self.cfg["training"]["gradient_clip"]))
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 20 == 0:
                    train_iter.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for enc, labels in tqdm(val_loader, desc=f"initial-val epoch {epoch + 1}", leave=False):
                    enc = {k: v.to(device) for k, v in enc.items()}
                    preds = model(**enc).logits.argmax(dim=-1).cpu().tolist()
                    y_pred.extend(preds)
                    y_true.extend(labels.tolist())

            val_acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true)))
            print(f"[InitialTrainer] epoch={epoch + 1} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                bad_epochs = 0
                model.save_pretrained(str(out_dir))
                tok.save_pretrained(str(out_dir))
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
        print(f"[InitialTrainer] best_val_acc={best_val:.4f}")
