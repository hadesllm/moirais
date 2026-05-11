"""Tests for morie.fn.stckp -- Stick-breaking."""

from morie.fn.stckp import stick_breaking


def test_returns_dict():
    result = stick_breaking(1.0)
    assert isinstance(result, dict)
    assert "weights" in result


def test_weights_sum_near_one():
    result = stick_breaking(1.0, K=100)
    assert abs(result["cumulative_weight"] - 1.0) < 0.05


def test_more_effective_with_low_alpha():
    r1 = stick_breaking(0.1, K=50)
    r2 = stick_breaking(10.0, K=50)
    assert r1["n_effective"] <= r2["n_effective"]
