import argparse
import copy
import gc
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from driftllm.trainers.baseline_trainer import run_baselines
from driftllm.trainers.initial_trainer import InitialTrainer
from driftllm.trainers.online_trainer import OnlineDriftTrainer
from driftllm.utils.config import load_config
from driftllm.utils.reproducibility import seed_everything


def _parse_value(v: str):
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _set_nested(cfg, key_path: str, value):
    parts = key_path.split(".")
    ref = cfg
    for p in parts[:-1]:
        if p not in ref or not isinstance(ref[p], dict):
            ref[p] = {}
        ref = ref[p]
    ref[parts[-1]] = value


def _maybe_init_wandb(cfg, method: str, seed: int):
    if not cfg["experiment"].get("use_wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['experiment'].get('domain', 'domain')}_{method}_{seed}_{ts}"
    return wandb.init(
        project=cfg["experiment"].get("wandb_project", "driftllm"),
        entity=cfg["experiment"].get("wandb_entity", None),
        name=run_name,
        config=cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="driftllm/configs/config.yaml")
    parser.add_argument("--mode", type=str, choices=["data", "train", "eval", "full"], default="full")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        choices=["selective", "no_update", "full_lora", "oracle", "no_ewc"],
        default="selective",
        help="Run one method at a time for reproducible experiment matrices.",
    )
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--set", action="append", default=[], help="Override config using key=value (supports nested keys).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        _set_nested(cfg, k, _parse_value(v))

    seeds_raw = cfg["experiment"].get("seeds", [cfg["experiment"]["seed"]])
    if isinstance(seeds_raw, (int, float, str)):
        seeds = [int(seeds_raw)]
    else:
        seeds = [int(s) for s in seeds_raw]
    runs = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["experiment"]["seed"] = int(seed)
        seed_everything(int(seed))
        if args.mode in {"train", "full"}:
            initial_ckpt = Path(run_cfg["paths"]["model_dir"]) / "initial"
            if initial_ckpt.exists():
                print(f"[main] Skipping initial training — checkpoint found at {initial_ckpt}")
            else:
                InitialTrainer(run_cfg).run()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        method = "baseline" if args.baseline else args.method
        wb = _maybe_init_wandb(run_cfg, method=method, seed=int(seed))
        if args.baseline:
            result = run_baselines(run_cfg)
        else:
            trainer_mode = {
                "selective": "selective",
                "no_update": "no_update",
                "full_lora": "full_retrain",
                "oracle": "oracle",
                "no_ewc": "selective",
            }[args.method]
            if args.method == "no_ewc":
                run_cfg["forgetting"]["ewc_lambda"] = 0.0
            result = OnlineDriftTrainer(run_cfg, mode=trainer_mode).run()
            result["method"] = args.method
        if wb is not None:
            try:
                wb.log({k: v for k, v in result.items() if isinstance(v, (int, float))})
                wb.finish()
            except Exception:
                pass
        runs.append({"seed": int(seed), "result": result})

    # Aggregate only top-level numeric metrics when available.
    agg = {}
    if runs and not args.baseline:
        keys = [k for k, v in runs[0]["result"].items() if isinstance(v, (int, float))]
        for k in keys:
            vals = [float(r["result"][k]) for r in runs]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    elif runs and args.baseline:
        methods = runs[0]["result"].keys()
        for method in methods:
            first = runs[0]["result"][method]
            keys = [k for k, v in first.items() if isinstance(v, (int, float))]
            agg[method] = {}
            for k in keys:
                vals = [float(r["result"][method][k]) for r in runs]
                agg[method][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    results = {"runs": runs, "aggregate": agg}

    if args.mode in {"eval", "full"}:
        out = Path(cfg["paths"]["results_dir"])
        out.mkdir(parents=True, exist_ok=True)
        cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        out_file = out / f"run_results_{args.tag}_{cfg_hash}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
