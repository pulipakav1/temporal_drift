#!/usr/bin/env bash
set -euo pipefail
python main.py --config driftllm/configs/config.yaml --mode eval --baseline --tag baselines_financial --set experiment.domain=financial
