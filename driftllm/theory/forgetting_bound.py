from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_theoretical_bound(fisher_diag, delta_theta, ewc_lambda, selected_layers):
    b = 0.0
    for n in selected_layers:
        fi = fisher_diag.get(n, 0.0)
        dt = delta_theta.get(n, 0.0)
        b += fi * (dt**2)
    return ewc_lambda * b


def verify_bound_empirically(theoretical_bound, actual_forgetting_measurements, epsilon=1e-6):
    arr = np.array(actual_forgetting_measurements)
    return bool(np.all(arr <= theoretical_bound + epsilon))


def plot_bound_vs_empirical(measurements, bounds, output_path):
    plt.figure(figsize=(5, 4))
    plt.scatter(bounds, measurements, alpha=0.8)
    m = max(max(bounds), max(measurements)) if bounds and measurements else 1.0
    plt.plot([0, m], [0, m], linestyle="--")
    plt.xlabel("Theoretical bound")
    plt.ylabel("Empirical forgetting")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
