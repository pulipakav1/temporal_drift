import argparse
import shlex
from pathlib import Path


SEEDS = [42, 52, 62]
DOMAINS = ["financial", "clinical"]

EXPERIMENTS = [
    {"name": "selective_ewc", "method": "selective", "sets": []},
    {"name": "no_update", "method": "no_update", "sets": []},
    {"name": "full_lora", "method": "full_lora", "sets": []},
    {"name": "oracle", "method": "oracle", "sets": []},
    {"name": "no_ewc", "method": "no_ewc", "sets": []},
    {"name": "topk4", "method": "selective", "sets": ["layer_selection.top_k=4"]},
    {"name": "topk16", "method": "selective", "sets": ["layer_selection.top_k=16"]},
    {"name": "replay128", "method": "selective", "sets": ["forgetting.replay_buffer=128"]},
    {"name": "replay0", "method": "selective", "sets": ["forgetting.replay_buffer=0"]},
    {"name": "mmd002", "method": "selective", "sets": ["drift.mmd_threshold=0.02"]},
]


def shell_join(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def build_command(args, domain, seed, exp):
    tag = f"{domain}_{exp['name']}_seed{seed}"
    results_dir = f"{args.results_root}/results/{domain}/{exp['name']}/seed{seed}"
    model_dir = f"{args.results_root}/models/{domain}/{exp['name']}/seed{seed}"
    plots_dir = f"{args.results_root}/plots"
    data_root = f"{args.results_root}/data"

    parts = [
        "accelerate",
        "launch",
        "--num_processes",
        args.num_processes,
        "--num_machines",
        "1",
        "--mixed_precision",
        args.mixed_precision,
        "--dynamo_backend",
        "no",
        "main.py",
        "--config",
        "driftllm/configs/config.yaml",
        "--mode",
        args.mode,
        "--method",
        exp["method"],
        "--tag",
        tag,
        "--set",
        f"experiment.domain={domain}",
        "--set",
        f"experiment.seeds={seed}",
        "--set",
        f"training.initial_epochs={args.initial_epochs}",
        "--set",
        f"training.adaptation_steps={args.adaptation_steps}",
        "--set",
        f"layer_selection.fisher_samples={args.fisher_samples}",
        "--set",
        f"paths.results_dir={results_dir}",
        "--set",
        f"paths.model_dir={model_dir}",
        "--set",
        f"paths.data_root={data_root}",
        "--set",
        f"paths.financial_cache={data_root}/financial_hf",
        "--set",
        f"paths.mimic_path={data_root}/mimic_iii.csv",
        "--set",
        f"paths.plots_dir={plots_dir}",
    ]
    for item in exp["sets"]:
        parts.extend(["--set", item])
    return shell_join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate the 60-run DriftLLM experiment matrix.")
    parser.add_argument("--output", default="scripts/run_matrix_60.sh")
    parser.add_argument("--results-root", default="/content/drive/MyDrive/temporal_drift_results")
    parser.add_argument("--num-processes", default="1")
    parser.add_argument("--mixed-precision", default="no")
    parser.add_argument("--mode", default="full", choices=["train", "eval", "full"])
    parser.add_argument("--initial-epochs", default="1")
    parser.add_argument("--adaptation-steps", default="10")
    parser.add_argument("--fisher-samples", default="32")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    args = parser.parse_args()

    commands = []
    for domain in args.domains:
        for exp in EXPERIMENTS:
            for seed in SEEDS:
                commands.append(build_command(args, domain, seed, exp))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        for command in commands:
            f.write(command + "\n")

    print(f"Wrote {len(commands)} commands to {out}")


if __name__ == "__main__":
    main()
