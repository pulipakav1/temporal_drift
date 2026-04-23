from pathlib import Path

import torch

from driftllm.data.builders import build_domain_splits
from driftllm.models.base_model import load_model_tokenizer


class InitialTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        splits = build_domain_splits(self.cfg)
        n_labels = self.cfg["model"]["num_labels_financial"] if self.cfg["experiment"]["domain"] == "financial" else self.cfg["model"]["num_labels_clinical"]
        model, tok = load_model_tokenizer(
            self.cfg["model"]["name"], n_labels, self.cfg["model"]["bf16"], self.cfg["model"]["device_map"]
        )
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(target_device)
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in ("score", "classifier"))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        device = next(model.parameters()).device
        train_rows = [dict(x) for x in splits["train"]]
        val_rows = [dict(x) for x in splits["validation"]]
        opt = torch.optim.AdamW(trainable_params, lr=self.cfg["training"]["lr"])
        patience = int(self.cfg["training"]["early_stopping_patience"])
        best_val = -1.0
        bad_epochs = 0
        out_dir = Path(self.cfg["paths"]["model_dir"]) / "initial"
        out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(int(self.cfg["training"].get("initial_epochs", 3))):
            model.train()
            for row in train_rows:
                enc = tok(
                    str(row["text"]), truncation=True, max_length=256, padding="max_length", return_tensors="pt"
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = torch.tensor([int(row["label"])], device=device)
                loss = model(**enc, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, float(self.cfg["training"]["gradient_clip"]))
                opt.step()
                opt.zero_grad(set_to_none=True)
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for row in val_rows:
                    enc = tok(
                        str(row["text"]), truncation=True, max_length=256, padding="max_length", return_tensors="pt"
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    pred = int(model(**enc).logits.argmax(dim=-1).item())
                    y_true.append(int(row["label"]))
                    y_pred.append(pred)
            val_acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true)))
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
