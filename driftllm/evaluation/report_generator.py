import json
from pathlib import Path


def save_report(report: dict, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def load_run_reports(results_dir: str):
    root = Path(results_dir)
    files = sorted(root.glob("run_results_*.json"))
    out = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def build_canonical_summary(run_reports):
    summary = {"n_reports": len(run_reports), "tags": [], "aggregates": []}
    for r in run_reports:
        summary["aggregates"].append(r.get("aggregate", {}))
    return summary
