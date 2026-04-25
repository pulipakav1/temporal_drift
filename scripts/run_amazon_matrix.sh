#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 52 62)
METHODS=(selective no_update full_lora oracle no_ewc)

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    python main.py \
      --config driftllm/configs/config.yaml \
      --mode full \
      --method "${method}" \
      --tag "amazon_${method}_seed${seed}" \
      --set experiment.domain=amazon \
      --set experiment.seeds="${seed}" \
      --set drift.mmd_threshold=0.04 \
      --set drift.adwin_delta=0.002 \
      --set drift.window_size=400 \
      --set drift.reference_size=800 \
      --set drift.cooldown=300 \
      --set drift.check_every_steps=50
  done
done
