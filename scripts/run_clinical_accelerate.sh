#!/usr/bin/env bash
set -euo pipefail

accelerate launch --num_processes 4 main.py \
  --config driftllm/configs/config.yaml \
  --mode full \
  --tag clinical_full_accelerate \
  --set experiment.domain=clinical
