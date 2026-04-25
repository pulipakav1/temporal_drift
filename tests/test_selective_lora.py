import pytest

from driftllm.evaluation.forgetting_evaluator import mean_forgetting
from driftllm.models.layer_sensitivity import LayerSensitivityAnalyzer


def test_layer_selection_semantic():
    analyzer = LayerSensitivityAnalyzer(top_k=2, min_score=0.01)
    fisher = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": 0.3,
        "model.layers.0.self_attn.v_proj.lora_A.weight": 0.2,
        "model.layers.0.mlp.down_proj.lora_A.weight": 0.9,
    }
    selected = analyzer.select_layers(fisher, "semantic_drift")
    assert len(selected) == 2
    assert all(("q_proj" in x or "v_proj" in x) for x in selected)


def test_mean_forgetting_helper():
    assert mean_forgetting([0.1, 0.0, 0.2]) == pytest.approx(0.1)


def test_layer_selection_random_route_k():
    analyzer = LayerSensitivityAnalyzer(top_k=2, min_score=0.0, routing_strategy="random", seed=7)
    fisher = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": 0.3,
        "model.layers.0.self_attn.v_proj.lora_A.weight": 0.2,
        "model.layers.0.mlp.down_proj.lora_A.weight": 0.9,
    }
    selected = analyzer.select_layers(fisher, "semantic_drift")
    assert len(selected) == 2
