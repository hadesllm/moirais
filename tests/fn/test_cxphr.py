"""Tests for cxphr: Cox proportional hazards via partial likelihood."""
import numpy as np
import pytest
from moirais.fn.cxphr import cxphr


RNG = np.random.default_rng(42)


def _make_data(n=100, beta_true=0.7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    # Exponential survival with hazard exp(beta * X)
    hazard = np.exp(beta_true * X[:, 0])
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event, X


def test_returns_expected_keys():
    time, event, X = _make_data(n=80)
    result = cxphr(time, event, X)
    for key in ("beta", "se", "z", "p_value", "log_likelihood", "converged"):
        assert key in result


def test_beta_shape():
    time, event, X = _make_data(n=80)
    result = cxphr(time, event, X)
    assert result["beta"].shape == (1,)
    assert result["se"].shape == (1,)


def test_correct_sign_of_beta():
    """Positive beta_true should give positive beta estimate."""
    time, event, X = _make_data(n=200, beta_true=1.0, seed=1)
    result = cxphr(time, event, X)
    assert result["beta"][0] > 0


def test_negative_beta():
    """Negative beta_true should give negative beta estimate."""
    time, event, X = _make_data(n=200, beta_true=-1.0, seed=2)
    result = cxphr(time, event, X)
    assert result["beta"][0] < 0


def test_se_positive():
    time, event, X = _make_data(n=100)
    result = cxphr(time, event, X)
    assert np.all(result["se"] > 0)


def test_p_value_range():
    time, event, X = _make_data(n=100)
    result = cxphr(time, event, X)
    assert np.all(result["p_value"] >= 0)
    assert np.all(result["p_value"] <= 1)


def test_null_beta_near_zero_under_no_effect():
    """Beta should be near 0 when there is no effect."""
    rng = np.random.default_rng(99)
    n = 300
    X = rng.standard_normal((n, 1))
    time = rng.exponential(1.0, size=n)  # constant hazard, no covariate effect
    event = np.ones(n)
    result = cxphr(time, event, X)
    assert abs(result["beta"][0]) < 0.5


def test_multivariate():
    rng = np.random.default_rng(7)
    n = 150
    X = rng.standard_normal((n, 3))
    beta_true = np.array([1.0, -0.5, 0.2])
    hazard = np.exp(X @ beta_true)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(2.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    result = cxphr(time, event, X)
    assert result["beta"].shape == (3,)
    assert result["se"].shape == (3,)
    # Signs should match
    assert result["beta"][0] > 0
    assert result["beta"][1] < 0


def test_efron_ties():
    time, event, X = _make_data(n=80)
    result = cxphr(time, event, X, ties="efron")
    assert "beta" in result


def test_invalid_ties_raises():
    time, event, X = _make_data(n=50)
    with pytest.raises(ValueError, match="ties"):
        cxphr(time, event, X, ties="bogus")


def test_negative_time_raises():
    time = np.array([-1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    X = np.array([[1.0], [0.0], [-1.0]])
    with pytest.raises(ValueError):
        cxphr(time, event, X)


def test_log_likelihood_is_finite():
    time, event, X = _make_data(n=100)
    result = cxphr(time, event, X)
    assert np.isfinite(result["log_likelihood"])
