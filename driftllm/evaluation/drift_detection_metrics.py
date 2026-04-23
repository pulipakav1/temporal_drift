def precision_recall_f1(tp: int, fp: int, fn: int):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-8, p + r)
    return {"precision": p, "recall": r, "f1": f1}


def deduplicate_contiguous_ground_truth_events(ground_truth_events, min_separation_steps: int = 25):
    """
    Collapse repeated per-sample event markings into one event onset.
    Input: list[{"step": int, "drift_type": str, ...}]
    """
    if not ground_truth_events:
        return []
    events = sorted(ground_truth_events, key=lambda x: int(x["step"]))
    dedup = [events[0]]
    for cur in events[1:]:
        prev = dedup[-1]
        same_type = cur["drift_type"] == prev["drift_type"]
        close = (int(cur["step"]) - int(prev["step"])) <= min_separation_steps
        if same_type and close:
            continue
        dedup.append(cur)
    return dedup


def detection_metrics_from_events(detected_events, ground_truth_events, tolerance_steps: int = 100):
    """
    detected_events: list[{"step": int, "drift_type": str}]
    ground_truth_events: list[{"step": int, "drift_type": str}]
    """
    matched_gt = set()
    tp, fp = 0, 0
    delays = []
    for det in detected_events:
        det_step = int(det["step"])
        det_type = det["drift_type"]
        best_idx = -1
        best_delay = None
        for i, gt in enumerate(ground_truth_events):
            if i in matched_gt or gt["drift_type"] != det_type:
                continue
            delay = det_step - int(gt["step"])
            if 0 <= delay <= tolerance_steps:
                if best_delay is None or delay < best_delay:
                    best_delay = delay
                    best_idx = i
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
            delays.append(best_delay)
        else:
            fp += 1
    fn = len(ground_truth_events) - len(matched_gt)
    base = precision_recall_f1(tp, fp, fn)
    base["mean_detection_delay_steps"] = float(sum(delays) / len(delays)) if delays else -1.0
    base["tp"] = int(tp)
    base["fp"] = int(fp)
    base["fn"] = int(fn)
    return base


def per_type_detection_metrics(detected_events, ground_truth_events, drift_types, tolerance_steps: int = 100):
    out = {}
    for d_type in drift_types:
        det = [x for x in detected_events if x["drift_type"] == d_type]
        gt = [x for x in ground_truth_events if x["drift_type"] == d_type]
        out[d_type] = detection_metrics_from_events(det, gt, tolerance_steps=tolerance_steps)
    return out
