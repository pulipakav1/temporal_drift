import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model

from driftllm.models.base_model import load_model_tokenizer
from driftllm.models.layer_sensitivity import LayerSensitivityAnalyzer


class SelectiveLoRAModel:
    def __init__(self, cfg, forgetting_reg):
        self.cfg = cfg
        self.forgetting_reg = forgetting_reg
        self.analyzer = LayerSensitivityAnalyzer(
            top_k=cfg["layer_selection"]["top_k"], min_score=cfg["layer_selection"]["min_fisher_score"]
        )
        self.model = None
        self.tokenizer = None

    def load(self, num_labels: int):
        initial_dir = Path(self.cfg["paths"]["model_dir"]) / "initial"
        checkpoint_dir = str(initial_dir) if initial_dir.exists() else None
        self.model, self.tokenizer = load_model_tokenizer(
            self.cfg["model"]["name"],
            num_labels,
            self.cfg["model"]["bf16"],
            self.cfg["model"]["device_map"],
            materialize_for_peft=True,
            checkpoint_dir=checkpoint_dir,
        )
        lc = self.cfg["lora"]
        lora_cfg = LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(target_device)
        return self

    def freeze_all_lora(self):
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = False

    def unfreeze_selected_layers(self, names):
        self.freeze_all_lora()
        c = 0
        n_params = 0
        for n, p in self.model.named_parameters():
            if "lora_" in n and any(sel in n for sel in names):
                p.requires_grad = True
                c += 1
                n_params += p.numel()
        return c, n_params

    @torch.no_grad()
    def get_embedding(self, input_ids, attention_mask, layer_idx=-1):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
        hs = out.hidden_states[layer_idx]
        mask = attention_mask.unsqueeze(-1)
        return (hs * mask).sum(1) / mask.sum(1).clamp_min(1)

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits, logits.argmax(dim=-1)

    def adapt_to_drift(self, drift_event, dataloader, n_steps):
        t0 = time.time()
        device = next(self.model.parameters()).device
        self.forgetting_reg.consolidate(self.model, dataloader, device)
        fisher = self.analyzer.compute_fisher(
            self.model, dataloader, n_samples=self.cfg["layer_selection"]["fisher_samples"]
        )
        layers = self.analyzer.select_layers(fisher, drift_event.drift_type)
        n_upd, n_upd_params = self.unfreeze_selected_layers(layers)
        total_lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return {
                "timing": time.time() - t0,
                "avg_loss": 0.0,
                "n_layers_updated": 0,
                "n_params_updated": 0,
                "pct_lora_params_updated": 0.0,
            }
        opt = torch.optim.AdamW(trainable_params, lr=self.cfg["training"]["lr"])
        losses = []
        for step, batch in enumerate(dataloader):
            if step >= n_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self.model(**batch)
            ewc = self.forgetting_reg.ewc_loss(self.model)
            replay_size = min(int(batch["labels"].shape[0]), len(self.forgetting_reg.replay_buffer))
            replay_batch = self.forgetting_reg.get_replay_batch(batch_size=replay_size)
            replay_loss = 0.0
            if replay_batch is not None:
                replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                replay_loss = self.model(**replay_batch).loss
            loss = out.loss + ewc + 0.3 * replay_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        self.freeze_all_lora()
        return {
            "timing": time.time() - t0,
            "avg_loss": sum(losses) / max(1, len(losses)),
            "n_layers_updated": n_upd,
            "n_params_updated": int(n_upd_params),
            "pct_lora_params_updated": float(n_upd_params / max(1, total_lora_params)),
        }
