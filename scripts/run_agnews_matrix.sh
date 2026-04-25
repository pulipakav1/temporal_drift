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
      --tag "agnews_${method}_seed${seed}" \
      --set experiment.domain=agnews \
      --set experiment.seeds="${seed}" \
      --set drift.mmd_threshold=0.03 \
      --set drift.adwin_delta=0.001 \
      --set drift.window_size=300 \
      --set drift.reference_size=600 \
      --set drift.cooldown=250 \
      --set drift.check_every_steps=40
  done
done
