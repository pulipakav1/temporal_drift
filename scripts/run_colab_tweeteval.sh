#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly single-GPU run:
# - uses TweetEval
# - streams validation+test (captures more drift events)
# - writes artifacts to Google Drive if mounted at /content/drive

RESULTS_ROOT="${RESULTS_ROOT:-/content/drive/MyDrive/temporal_drift_results}"
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
SEED="${SEED:-42}"
TAG="${TAG:-colab_tweeteval_selective}"

mkdir -p "${RESULTS_ROOT}/data" "${RESULTS_ROOT}/models" "${RESULTS_ROOT}/results" "${RESULTS_ROOT}/plots"

python main.py \
  --config driftllm/configs/config.yaml \
  --mode full \
  --method selective \
  --tag "${TAG}" \
  --set experiment.domain=tweeteval \
  --set experiment.seed="${SEED}" \
  --set experiment.seeds="${SEED}" \
  --set experiment.stream_split=validation_test \
  --set model.name="${MODEL_NAME}" \
  --set model.device_map=auto \
  --set model.bf16=false \
  --set training.initial_epochs=1 \
  --set training.adaptation_steps=10 \
  --set layer_selection.fisher_samples=32 \
  --set training.batch_size=8 \
  --set training.grad_accum=2 \
  --set paths.data_root="${RESULTS_ROOT}/data" \
  --set paths.financial_cache="${RESULTS_ROOT}/data/financial_hf" \
  --set paths.tweet_cache="${RESULTS_ROOT}/data/tweeteval_hf" \
  --set paths.mimic_path="${RESULTS_ROOT}/data/mimic_iii.csv" \
  --set paths.model_dir="${RESULTS_ROOT}/models" \
  --set paths.results_dir="${RESULTS_ROOT}/results" \
  --set paths.plots_dir="${RESULTS_ROOT}/plots"

echo "Run complete. Results under ${RESULTS_ROOT}/results"
