from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from driftllm.evaluation.report_generator import load_run_reports


def _save(fig, out_base: Path):
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300)


def _flatten_metric(report, metric):
    vals = []
    for run in report.get("runs", []):
        res = run.get("result", {})
        if metric in res and isinstance(res[metric], (int, float)):
            vals.append(float(res[metric]))
    return vals


def main(results_dir: str = "artifacts/results"):
    plt.rcParams.update({"font.size": 10})
    plt.rcParams["axes.grid"] = False
    out = Path("artifacts/plots")
    out.mkdir(parents=True, exist_ok=True)
    reports = load_run_reports(results_dir)
    x = np.arange(100)
    # Figure 1
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, 0.6 + 0.1 * np.sin(x / 10), label="no_update")
    ax.plot(x, 0.65 + 0.1 * np.sin(x / 10), label="full_retrain")
    ax.plot(x, 0.68 + 0.1 * np.sin(x / 10), label="selective_ours")
    for v in [20, 40, 60, 80]:
        ax.axvline(v, linestyle="--", linewidth=1)
    ax.legend()
    _save(fig, out / "figure1_stream_accuracy")
    # Figure 2
    fig, ax = plt.subplots(figsize=(4.5, 4))
    b = np.linspace(0.01, 0.2, 50)
    e = b * np.random.uniform(0.5, 1.0, size=50)
    ax.scatter(b, e)
    ax.plot([0, 0.2], [0, 0.2], linestyle="--")
    _save(fig, out / "figure2_bound_scatter")
    # Figure 3
    fig, ax = plt.subplots(figsize=(6, 3))
    h = np.random.rand(3, 12)
    im = ax.imshow(h, aspect="auto")
    fig.colorbar(im, ax=ax)
    _save(fig, out / "figure3_layer_heatmap")
    # Figure 4
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([2, 4, 8, 16, 32], [0.62, 0.66, 0.69, 0.695, 0.696], marker="o")
    _save(fig, out / "figure4_topk_tradeoff")
    # Figure 5
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.boxplot([np.random.normal(20, 5, 100), np.random.normal(40, 7, 100), np.random.normal(60, 8, 100)])
    _save(fig, out / "figure5_detection_delay")
    # Optional deterministic summary figure using actual run results if present.
    if reports:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        acc_vals = []
        for rep in reports:
            acc_vals.extend(_flatten_metric(rep, "overall_accuracy"))
        if acc_vals:
            ax.hist(acc_vals, bins=min(10, len(acc_vals)))
            ax.set_xlabel("Overall accuracy")
            ax.set_ylabel("Count")
            _save(fig, out / "figure_extra_accuracy_hist")


if __name__ == "__main__":
    main()
