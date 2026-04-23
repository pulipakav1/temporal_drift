from driftllm.theory.forgetting_bound import compute_theoretical_bound, verify_bound_empirically


def test_bound():
    b = compute_theoretical_bound({"a": 2.0}, {"a": 0.5}, 0.5, ["a"])
    assert b > 0
    assert verify_bound_empirically(b, [0.01, b - 1e-4])


def test_bound_violation():
    b = compute_theoretical_bound({"x": 1.0}, {"x": 0.1}, 0.5, ["x"])
    assert not verify_bound_empirically(b, [b + 0.5], epsilon=1e-8)
