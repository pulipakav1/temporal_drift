import argparse
import os
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--domain", choices=["tweeteval", "amazon", "all"], default="all")
parser.add_argument("--results_root", default=None)
args = parser.parse_args()

# ---------- Config ----------
RESULTS_ROOT = args.results_root or os.environ.get(
    "RESULTS_ROOT", "/scratch/results/temporal_drift"
)

COMMON = [
    "--config", "driftllm/configs/config.yaml",
    "--mode", "full",
    "--set", "experiment.stream_split=validation_test",
    "--set", "model.name=Qwen/Qwen2.5-7B-Instruct",
    "--set", "model.bf16=true",
    "--set", "training.initial_epochs=3",
    "--set", f"paths.model_dir={RESULTS_ROOT}/models",
    "--set", f"paths.results_dir={RESULTS_ROOT}/results",
    "--set", f"paths.plots_dir={RESULTS_ROOT}/plots",
    "--set", f"paths.tweet_cache={RESULTS_ROOT}/data/tweeteval_hf",
    "--set", f"paths.amazon_cache={RESULTS_ROOT}/data/amazon_hf",
]

FAST_ADAPT = [
    "--set", "training.adaptation_steps=30",
    "--set", "layer_selection.fisher_samples=64",
    "--set", "forgetting.replay_buffer=256",
]

SEEDS = [42, 52, 62]

ALL_RUNS = [
    ("tweeteval", "selective"),
    ("tweeteval", "no_update"),
    ("tweeteval", "full_lora"),
    ("amazon",    "selective"),
    ("amazon",    "no_update"),
    ("amazon",    "full_lora"),
]

RUNS = ALL_RUNS if args.domain == "all" else [r for r in ALL_RUNS if r[0] == args.domain]
# ---------------------------


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s"


def eta(seconds_from_now):
    return datetime.fromtimestamp(time.time() + seconds_from_now).strftime("%H:%M:%S")


all_start = time.time()
durations = []
results = []

total = len(RUNS) * len(SEEDS)
domains = sorted({r[0] for r in RUNS})
print(f"Starting {total} runs ({len(RUNS)} configs × {len(SEEDS)} seeds)")
print(f"Domains: {', '.join(domains)}  |  Methods: selective, no_update, full_lora")
print(f"Results root: {RESULTS_ROOT}")
print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

run_idx = 0
for domain, method in RUNS:
    for seed in SEEDS:
        run_idx += 1
        tag = f"emnlp_{domain}_{method}_seed{seed}"

        cmd_parts = [
            "python", "main.py",
            "--method", method,
            "--tag", tag,
            "--set", f"experiment.domain={domain}",
            "--set", f"experiment.seeds={seed}",
        ] + COMMON

        if method in ("selective", "full_lora"):
            cmd_parts += FAST_ADAPT

        cmd_str = " ".join(cmd_parts)

        print(f"[{run_idx}/{total}] {tag}")
        start = time.time()
        code = os.system(cmd_str)
        elapsed = time.time() - start

        durations.append(elapsed)
        results.append((tag, code, elapsed))

        remaining = total - run_idx
        avg = sum(durations) / len(durations)
        status = "OK" if code == 0 else f"FAILED(code={code})"
        print(f"  -> {status} in {fmt(elapsed)}", end="")
        if remaining > 0:
            print(f"  |  avg {fmt(avg)}/run  |  ETA ~{fmt(remaining * avg)} ({eta(remaining * avg)})")
        else:
            print()

total_elapsed = time.time() - all_start

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
ok, fail = 0, 0
for tag, code, elapsed in results:
    status = "OK" if code == 0 else f"FAIL({code})"
    print(f"  {status:10s}  {fmt(elapsed):8s}  {tag}")
    if code == 0:
        ok += 1
    else:
        fail += 1
print("-" * 70)
print(f"  {ok}/{total} successful  |  Total: {fmt(total_elapsed)}")
print("=" * 70)
