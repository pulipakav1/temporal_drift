import os
import time
from datetime import datetime

# Local debug script — pipeline validation only, NOT paper results.
# Uses 1.5B model to fit in 6GB VRAM. For paper results use run_emnlp_tweeteval_amazon.py on A100.

COMMON = [
    "--config", "driftllm/configs/config.yaml",
    "--mode", "full",
    "--set", "experiment.stream_split=validation_test",
    "--set", "model.name=Qwen/Qwen2.5-1.5B-Instruct",
    "--set", "model.bf16=true",
    "--set", "training.initial_epochs=1",
    "--set", "experiment.seeds=42",
]

FAST_ADAPT = [
    "--set", "training.adaptation_steps=10",
    "--set", "layer_selection.fisher_samples=16",
    "--set", "forgetting.replay_buffer=64",
]

# Single seed, two domains, three methods — just enough to verify the pipeline end-to-end.
RUNS = [
    ("tweeteval", "selective"),
    ("tweeteval", "no_update"),
    ("tweeteval", "full_lora"),
    ("amazon",    "selective"),
    ("amazon",    "no_update"),
    ("amazon",    "full_lora"),
]


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s"


all_start = time.time()
results = []
durations = []

print(f"Local debug run — Qwen2.5-1.5B on RTX 3050 6GB")
print(f"Results → ./artifacts/results/")
print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for i, (domain, method) in enumerate(RUNS, 1):
    tag = f"local_{domain}_{method}_seed42"
    cmd = " ".join([
        "python", "main.py",
        "--method", method,
        "--tag", tag,
        "--set", f"experiment.domain={domain}",
        "--set", "experiment.seeds=42",
    ] + COMMON + (FAST_ADAPT if method in ("selective", "full_lora") else []))

    print(f"[{i}/{len(RUNS)}] {tag}")
    start = time.time()
    code = os.system(cmd)
    elapsed = time.time() - start
    durations.append(elapsed)
    results.append((tag, code, elapsed))

    status = "OK" if code == 0 else f"FAILED({code})"
    remaining = len(RUNS) - i
    avg = sum(durations) / len(durations)
    print(f"  -> {status} in {fmt(elapsed)}" + (f"  |  ETA ~{fmt(remaining * avg)}" if remaining else "") + "\n")

print("=" * 60)
ok = sum(1 for _, c, _ in results if c == 0)
for tag, code, elapsed in results:
    print(f"  {'OK' if code == 0 else 'FAIL':4s}  {fmt(elapsed):8s}  {tag}")
print(f"\n{ok}/{len(results)} passed  |  total {fmt(time.time() - all_start)}")
print("=" * 60)
