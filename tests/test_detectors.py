import torch

from driftllm.detectors.semantic_detector import SemanticDriftDetector
from driftllm.evaluation.drift_detection_metrics import deduplicate_contiguous_ground_truth_events


def test_semantic_detector_runs():
    d = SemanticDriftDetector(threshold=0.01, ref_size=50, window_size=50)
    event = None
    for i in range(200):
        event = d.update(torch.randn(4096), i + 1)
    assert event is None or event.drift_type == "semantic_drift"


def test_gt_event_dedup():
    events = [
        {"step": 10, "drift_type": "label_drift"},
        {"step": 11, "drift_type": "label_drift"},
        {"step": 40, "drift_type": "label_drift"},
        {"step": 43, "drift_type": "semantic_drift"},
    ]
    dedup = deduplicate_contiguous_ground_truth_events(events, min_separation_steps=5)
    assert len(dedup) == 3
