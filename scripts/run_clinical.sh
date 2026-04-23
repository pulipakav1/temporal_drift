#!/usr/bin/env bash
set -euo pipefail
python main.py --config driftllm/configs/config.yaml --mode full --tag clinical_full --set experiment.domain=clinical
