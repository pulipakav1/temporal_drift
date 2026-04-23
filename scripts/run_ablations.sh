#!/usr/bin/env bash
set -euo pipefail

TOP_KS=(2 4 8 16 32)
LAMBDAS=(0.0 0.1 0.5 1.0 2.0)
MMDS=(0.01 0.02 0.05 0.10)

for k in "${TOP_KS[@]}"; do
  python main.py --config driftllm/configs/config.yaml --mode full --tag "topk_${k}" --set layer_selection.top_k="${k}"
done

for l in "${LAMBDAS[@]}"; do
  python main.py --config driftllm/configs/config.yaml --mode full --tag "ewc_${l}" --set forgetting.ewc_lambda="${l}"
done

for m in "${MMDS[@]}"; do
  python main.py --config driftllm/configs/config.yaml --mode full --tag "mmd_${m}" --set drift.mmd_threshold="${m}"
done

# Drift-type routing ablation (all layers vs selective) and EWC ablation
python main.py --config driftllm/configs/config.yaml --mode full --tag "baseline_bundle" --baseline
python main.py --config driftllm/configs/config.yaml --mode full --tag "no_ewc" --set forgetting.ewc_lambda=0.0
