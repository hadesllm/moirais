"""Tests for cstat: Harrell's C-index."""
import numpy as np
import pytest
from morie.fn.cstat import cstat


def _make_data(n=100, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    T = rng.exponential(1.0 / np.exp(beta * X))
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    risk = np.exp(beta * X)
    return time, event, risk


def test_returns_keys():
    time, event, risk = _make_data()
    result = cstat(time, event, risk)
    for key in ("c_statistic", "se", "ci_lower", "ci_upper",
                "concordant", "discordant", "comparable"):
        assert key in result


def test_perfect_discrimination():
    """When risk perfectly predicts order, C should be close to 1."""
    n = 50
    time = np.arange(n, 0, -1, dtype=float)  # shorter time = higher rank
    event = np.ones(n)
    risk = np.arange(1, n + 1, dtype=float)  # higher risk = shorter time
    result = cstat(time, event, risk)
    assert result["c_statistic"] > 0.9


def test_random_prediction_near_0_5():
    """Random risk scores should give C near 0.5."""
    rng = np.random.default_rng(99)
    n = 200
    time = rng.exponential(1.0, size=n)
    event = np.ones(n)
    risk = rng.uniform(0, 1, size=n)
    result = cstat(time, event, risk)
    assert 0.3 < result["c_statistic"] < 0.7


def test_c_statistic_range():
    time, event, risk = _make_data(n=100, beta=1.0)
    result = cstat(time, event, risk)
    assert 0.0 <= result["c_statistic"] <= 1.0


def test_ci_contains_estimate():
    time, event, risk = _make_data()
    result = cstat(time, event, risk)
    assert result["ci_lower"] <= result["c_statistic"] <= result["ci_upper"]


def test_se_positive():
    time, event, risk = _make_data()
    result = cstat(time, event, risk)
    assert result["se"] >= 0


def test_concordant_plus_discordant_plus_tied_equals_comparable():
    time, event, risk = _make_data(n=80)
    result = cstat(time, event, risk)
    total = result["concordant"] + result["discordant"] + result["tied"]
    assert total == result["comparable"]


def test_uno_method():
    time, event, risk = _make_data(n=100, beta=1.0)
    result = cstat(time, event, risk, method="uno")
    assert 0.0 <= result["c_statistic"] <= 1.0


def test_invalid_method_raises():
    time, event, risk = _make_data()
    with pytest.raises(ValueError):
        cstat(time, event, risk, method="bogus")
