import json
from collections import deque
from pathlib import Path
from time import time

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from driftllm.data.builders import build_adaptation_loader, build_stream_loader
from driftllm.data.drift_annotations import event_name_to_type
from driftllm.detectors.knowledge_detector import KnowledgeDriftDetector
from driftllm.detectors.label_detector import LabelDriftDetector
from driftllm.detectors.orchestrator import DriftOrchestrator
from driftllm.detectors.semantic_detector import SemanticDriftDetector
from driftllm.evaluation.drift_detection_metrics import (
    deduplicate_contiguous_ground_truth_events,
    detection_metrics_from_events,
    per_type_detection_metrics,
)
from driftllm.evaluation.forgetting_evaluator import forgetting_delta, mean_forgetting
from driftllm.evaluation.recovery_speed import recovery_steps_to_fraction
from driftllm.models.forgetting_regularizer import ForgettingRegularizer
from driftllm.models.selective_lora import SelectiveLoRAModel
from driftllm.theory.forgetting_bound import compute_theoretical_bound


class OnlineDriftTrainer:
    def __init__(self, cfg, mode: str = "selective"):
        self.cfg = cfg
        self.mode = mode
        self.forgetting = ForgettingRegularizer(cfg["forgetting"]["ewc_lambda"], cfg["forgetting"]["replay_buffer"])
        if cfg["experiment"]["domain"] == "financial":
            n_labels = 3
        elif cfg["experiment"]["domain"] == "arxiv":
            n_labels = int(cfg["model"]["num_labels_arxiv"])
        else:
            n_labels = int(cfg["model"]["num_labels_clinical"])
        self.model = SelectiveLoRAModel(cfg, self.forgetting).load(n_labels)
        self.orch = DriftOrchestrator(
            SemanticDriftDetector(cfg["drift"]["mmd_threshold"], cfg["drift"]["reference_size"], cfg["drift"]["window_size"]),
            LabelDriftDetector(n_labels, cfg["drift"]["adwin_delta"]),
            KnowledgeDriftDetector(cfg["experiment"]["domain"]),
            cfg["drift"]["cooldown"],
        )
        self.metrics = {"drift_events": []}
        self.recent_true, self.recent_pred = deque(maxlen=100), deque(maxlen=100)
        self.buffer = deque(maxlen=cfg["forgetting"]["replay_buffer"])
        self.global_true = []
        self.global_pred = []
        self._drift_eval_buffer = deque(maxlen=100)
        self._gt_drift_events = []
        self._event_type_map = event_name_to_type(cfg["experiment"]["domain"])
        self._post_event_rolling_acc = []
        self.run_state = {"start_time": time(), "steps_completed": 0}
        self.ckpt_dir = Path(cfg["paths"]["results_dir"]) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _log_rolling_metrics(self, step):
        if step % 100 != 0 or len(self.recent_true) < 20:
            return
        y, p = list(self.recent_true), list(self.recent_pred)
        print(f"[online] step={step} rolling_acc={np.mean(np.array(y)==np.array(p)):.4f} rolling_f1={f1_score(y,p,average='macro'):.4f}")

    def _handle_drift_event(self, event):
        if self.mode == "no_update":
            return
        pre = float(np.mean(np.array(self.global_true[-100:]) == np.array(self.global_pred[-100:]))) if len(self.global_true) >= 20 else 0.0
        adapt_loader = build_adaptation_loader(list(self.buffer), batch_size=8)
        if adapt_loader is None:
            return
        before = {n: p.detach().clone() for n, p in self.model.model.named_parameters() if "lora_" in n}
        if self.mode == "full_retrain":
            old_top_k = self.model.analyzer.top_k
            self.model.analyzer.top_k = 10**9
            log = self.model.adapt_to_drift(event, adapt_loader, n_steps=int(self.cfg["training"].get("adaptation_steps", 50)))
            self.model.analyzer.top_k = old_top_k
        else:
            log = self.model.adapt_to_drift(event, adapt_loader, n_steps=int(self.cfg["training"].get("adaptation_steps", 50)))
        after = {n: p.detach().clone() for n, p in self.model.model.named_parameters() if "lora_" in n}
        fisher = self.forgetting.fisher_diag
        delta = {n: float((after[n] - before[n]).norm().item()) for n in after.keys() if n in before}
        selected_layers = list(delta.keys())
        bound = compute_theoretical_bound(
            {k: float(v.mean().item()) for k, v in fisher.items()},
            delta,
            self.cfg["forgetting"]["ewc_lambda"],
            selected_layers,
        )
        self.orch.semantic.update_reference()
        post = float(np.mean(np.array(self.global_true[-100:]) == np.array(self.global_pred[-100:]))) if len(self.global_true) >= 20 else pre
        f_delta = forgetting_delta(pre, post)
        rec = {
            "step": event.step,
            "drift_type": event.drift_type,
            "severity": event.severity,
            "forgetting_delta": f_delta,
            "theoretical_bound": float(bound),
            "bound_violated": bool(f_delta > (bound + 1e-6)),
            **log,
        }
        self.metrics["drift_events"].append(rec)

    def run(self, stream_loader=None):
        if stream_loader is None:
            stream_loader = build_stream_loader(self.cfg, self.model.tokenizer, split="test", batch_size=1)
        for step, batch in enumerate(stream_loader, start=1):
            try:
                device = next(self.model.model.parameters()).device
                model_batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
                _, preds = self.model.predict(model_batch["input_ids"], model_batch["attention_mask"])
                emb = self.model.get_embedding(model_batch["input_ids"], model_batch["attention_mask"])[0].detach().cpu()
                label = int(model_batch["labels"][0].item())
                pred = int(preds[0].item())
                self.recent_true.append(label)
                self.recent_pred.append(pred)
                self.global_true.append(label)
                self.global_pred.append(pred)
                sample = {k: v[0].detach().cpu() for k, v in model_batch.items()}
                self.forgetting.add_to_replay(sample)
                self.buffer.append(sample)
                self._drift_eval_buffer.append({"step": step, "pred": pred, "gt_event": batch["drift_event"][0]})
                if batch["drift_event"][0] != "none" and batch["drift_event"][0] in self._event_type_map:
                    self._gt_drift_events.append(
                        {"step": step, "drift_type": self._event_type_map[batch["drift_event"][0]], "name": batch["drift_event"][0]}
                    )
                ppl = None
                if step % 500 == 0:
                    ppl = self.orch.knowledge.compute_probe_perplexity(self.model.model, self.model.tokenizer, device)
                event = self.orch.update(step, embedding=emb, pred_label=pred, ppl=ppl)
                if self.mode == "oracle" and step % 200 == 0 and event is None:
                    from driftllm.detectors.base_detector import DriftEvent

                    event = DriftEvent(step=step, drift_type="knowledge_drift", score=1.0, threshold=0.0, severity=1.0, detector="oracle")
                if event:
                    self._handle_drift_event(event)
                    self._post_event_rolling_acc.append(float(np.mean(np.array(self.recent_true) == np.array(self.recent_pred))) if len(self.recent_true) > 0 else 0.0)
                self._log_rolling_metrics(step)
                self.run_state["steps_completed"] = step
                if step % int(self.cfg["training"].get("checkpoint_every_steps", 500)) == 0:
                    self._save_checkpoint(step)
            except Exception as exc:
                self._save_checkpoint(step, error=str(exc))
                raise
        return self._compile_results()

    def _save_checkpoint(self, step: int, error: str = ""):
        payload = {
            "step": step,
            "error": error,
            "n_events": len(self.metrics["drift_events"]),
            "n_predictions": len(self.global_pred),
        }
        with (self.ckpt_dir / f"online_step_{step}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _compile_results(self):
        y, p = self.global_true, self.global_pred
        f1_macro = float(f1_score(y, p, average="macro")) if y else 0.0
        _, _, f1_weighted, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0) if y else (0, 0, 0, 0)
        mean_forget = mean_forgetting([e.get("forgetting_delta", 0.0) for e in self.metrics["drift_events"]])
        total_adapt = float(np.sum([e.get("timing", 0.0) for e in self.metrics["drift_events"]])) if self.metrics["drift_events"] else 0.0
        n_param_updates = int(np.sum([e.get("n_layers_updated", 0) for e in self.metrics["drift_events"]])) if self.metrics["drift_events"] else 0
        gt_events = deduplicate_contiguous_ground_truth_events(self._gt_drift_events, min_separation_steps=25)
        det_metrics = detection_metrics_from_events(self.metrics["drift_events"], gt_events)
        per_type = per_type_detection_metrics(
            self.metrics["drift_events"],
            gt_events,
            drift_types=["semantic_drift", "label_drift", "knowledge_drift"],
        )
        bound_viol_rate = (
            float(np.mean([1.0 if e.get("bound_violated", False) else 0.0 for e in self.metrics["drift_events"]]))
            if self.metrics["drift_events"]
            else 0.0
        )
        pre_event_acc = float(np.mean(np.array(y[-100:]) == np.array(p[-100:]))) if len(y) >= 20 else 0.0
        recovery = recovery_steps_to_fraction(self._post_event_rolling_acc, pre_event_acc=pre_event_acc, fraction=0.9) if self._post_event_rolling_acc else -1
        res = {
            "method": self.mode,
            "overall_accuracy": float(np.mean(np.array(y) == np.array(p))) if y else 0.0,
            "f1_macro": f1_macro,
            "f1_weighted": float(f1_weighted),
            "drift_events": self.metrics["drift_events"],
            "avg_adaptation_loss": float(np.mean([e.get("avg_loss", 0.0) for e in self.metrics["drift_events"]])) if self.metrics["drift_events"] else 0.0,
            "avg_post_event_rolling_accuracy": float(np.mean(self._post_event_rolling_acc)) if self._post_event_rolling_acc else 0.0,
            "mean_forgetting": mean_forget,
            "n_param_updates": n_param_updates,
            "total_adapt_time_seconds": total_adapt,
            "drift_detection_precision": det_metrics["precision"],
            "drift_detection_recall": det_metrics["recall"],
            "drift_detection_f1": det_metrics["f1"],
            "drift_detection_tp": det_metrics["tp"],
            "drift_detection_fp": det_metrics["fp"],
            "drift_detection_fn": det_metrics["fn"],
            "mean_detection_delay_steps": det_metrics["mean_detection_delay_steps"],
            "drift_detection_per_type": per_type,
            "forgetting_bound_violation_rate": bound_viol_rate,
            "forgetting_bound_violation_count": int(sum(1 for e in self.metrics["drift_events"] if e.get("bound_violated", False))),
            "recovery_speed_steps_to_90pct": int(recovery),
        }
        out = Path(self.cfg["paths"]["results_dir"])
        out.mkdir(parents=True, exist_ok=True)
        with (out / "online_results.json").open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        return res
