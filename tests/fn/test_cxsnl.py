"""Tests for cxsnl: Cox-Snell residuals."""
import numpy as np
import pytest
from morie.fn.cxsnl import cxsnl


def _make_data(n=150, beta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    T = rng.exponential(1.0 / np.exp(beta * X[:, 0]))
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event, X, np.array([beta])


def test_returns_keys():
    time, event, X, beta = _make_data()
    result = cxsnl(time, event, X, beta)
    for key in ("residuals", "km_times", "km_neg_log_surv"):
        assert key in result


def test_residuals_nonneg():
    """Cox-Snell residuals are non-negative (they are cumulative hazards)."""
    time, event, X, beta = _make_data()
    result = cxsnl(time, event, X, beta)
    assert np.all(result["residuals"] >= 0)


def test_residuals_shape():
    time, event, X, beta = _make_data(n=100)
    result = cxsnl(time, event, X, beta)
    assert result["residuals"].shape == (100,)


def test_km_neg_log_surv_increasing():
    """KM of Cox-Snell residuals should have increasing -log(KM)."""
    time, event, X, beta = _make_data()
    result = cxsnl(time, event, X, beta)
    if len(result["km_neg_log_surv"]) > 1:
        assert np.all(np.diff(result["km_neg_log_surv"]) >= -1e-12)


def test_weibull_model():
    """Weibull model returns non-negative Cox-Snell residuals."""
    rng = np.random.default_rng(5)
    n = 100
    X = rng.standard_normal((n, 1))
    T = rng.weibull(1.5, size=n) * 2
    C = rng.exponential(4.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    beta = np.array([0.3])
    result = cxsnl(time, event, X, beta, model="weibull",
                   model_params={"log_lambda": -1.0, "rho": 1.5})
    assert np.all(result["residuals"] >= 0)


def test_invalid_model_raises():
    time, event, X, beta = _make_data()
    with pytest.raises(ValueError):
        cxsnl(time, event, X, beta, model="bogus")


def test_weibull_without_params_raises():
    time, event, X, beta = _make_data()
    with pytest.raises(ValueError):
        cxsnl(time, event, X, beta, model="weibull")
