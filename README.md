# DriftLLM: Drift-Type-Aware Selective LoRA Adaptation

DriftLLM is a research and deployment-oriented framework for detecting drift in streamed task inputs and performing selective LoRA adaptation with forgetting-aware regularization.

## Model Choice Justification: Qwen2.5-7B-Instruct

- Strong financial language understanding relative to similarly sized open alternatives.
- Better text-number tokenization behavior for tickers, rates, and mixed numeric phrases.
- Good LoRA sample efficiency for rapid post-drift adaptation.
- Stable BF16 execution in multi-GPU settings.
- Practical fit for 4 x A100 80GB setup with ZeRO-3 style distributed training.

## Implemented Contributions

- C1: Drift taxonomy and detectors:
  - `semantic_drift` (embedding-space MMD),
  - `label_drift` (ADWIN on label stream),
  - `knowledge_drift` (probe perplexity monitoring).
- C2: Fisher-guided layer sensitivity and selective adaptation routing.
- C3: EWC + replay regularization and theoretical forgetting bound checks.
- C4: DriftLLM-Eval scaffold (temporal splits, event annotations, streaming evaluation protocol).

## Environment Setup

```bash
pip install -r requirements.txt
```

For Google Colab, use the lean dependency set:

```bash
pip install -r requirements-colab.txt
```

## Reproducible Entry Points

- Financial full pipeline:
  - `bash scripts/run_financial.sh`
- Clinical full pipeline:
  - `bash scripts/run_clinical.sh`
- Baseline suite:
  - `bash scripts/run_baselines.sh`
- Ablations:
  - `bash scripts/run_ablations.sh`

Results are written to `artifacts/results/` with config-hashed filenames for reproducibility.

## Multi-GPU Launch

```bash
accelerate launch --num_processes 4 main.py --mode full --config driftllm/configs/config.yaml
```

Convenience scripts are also provided:

```bash
bash scripts/run_financial_accelerate.sh
bash scripts/run_clinical_accelerate.sh
```

## Core Metrics Reported

- Stream accuracy (rolling and overall),
- Macro and weighted F1,
- Drift detection precision/recall/F1 + delay,
- Forgetting delta and mean forgetting,
- Recovery speed (steps to 90% of pre-drift performance),
- Theoretical forgetting bound violation rate,
- Parameter/computation efficiency (`n_param_updates`, adaptation time).

## Reproducibility Protocol

- Fixed random seeding across `torch`, `numpy`, and `random`.
- Multi-seed execution via `experiment.seeds` in config.
- Config overrides supported through CLI:
  - `python main.py --config driftllm/configs/config.yaml --set layer_selection.top_k=16 --set forgetting.ewc_lambda=1.0`
- Optional Weights & Biases logging:
  - `--set experiment.use_wandb=true`
  - run naming follows `{domain}_{method}_{seed}_{timestamp}`.

## Dataset Notes

- Financial primary dataset loads from `takala/financial_phrasebank` (`sentences_allagree`) with pseudo-temporal assignment and event-specific drift injections.
- Public secondary dataset option is `cardiffnlp/tweet_eval` (`sentiment`) with pseudo-temporal assignment and event windows for major topic shifts.
- Clinical secondary dataset expects local MIMIC-style CSV configured via `paths.mimic_path`.
- MIMIC-III requires authorized access and proper credentialing; this repository does not ship restricted data.

## Paper Figure Generation

```bash
python scripts/plot_results.py
```

Saves publication figures in both PNG and PDF at 300 DPI under `artifacts/plots/`.

## Experiment Matrix

Generate the 60-run paper matrix:

```bash
python scripts/generate_run_matrix.py --output scripts/run_matrix_60.sh
```

The default matrix is:

- 2 domains: financial and tweeteval,
- 3 seeds: 42, 52, 62,
- 11 settings: selective EWC, no update, full LoRA, oracle, no EWC, top-k 4, top-k 16, replay 128, replay 0, MMD threshold 0.02, and random routing ablation.

Run the generated script from the repository root:

```bash
bash scripts/run_matrix_60.sh
```

Clinical runs require a MIMIC-style CSV at `paths.mimic_path` with note/timestamp/code columns.

## Colab Quickstart (TweetEval)

In Colab:

```bash
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone <YOUR_REPO_URL> temporal-drift
%cd /content/temporal-drift
!pip install -r requirements-colab.txt
!bash scripts/run_colab_tweeteval.sh
```

Optional overrides:

```bash
!RESULTS_ROOT=/content/drive/MyDrive/temporal_drift_results MODEL_NAME=distilbert-base-uncased TAG=tweeteval_run1 bash scripts/run_colab_tweeteval.sh
```
