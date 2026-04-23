import numpy as np


def summarize_fisher_distribution(fisher_dict):
    vals = np.array(list(fisher_dict.values())) if fisher_dict else np.array([0.0])
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p90": float(np.quantile(vals, 0.90)),
        "p99": float(np.quantile(vals, 0.99)),
    }


def fisher_by_module_type(fisher_dict):
    groups = {"attention": [], "ffn": [], "classifier": [], "other": []}
    for name, val in fisher_dict.items():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            groups["attention"].append(val)
        elif any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
            groups["ffn"].append(val)
        elif any(k in name for k in ["classifier", "score"]):
            groups["classifier"].append(val)
        else:
            groups["other"].append(val)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in groups.items()}
