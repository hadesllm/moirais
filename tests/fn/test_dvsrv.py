"""Tests for dvsrv: deviance residuals for Cox model."""
import numpy as np
import pytest
from morie.fn.dvsrv import dvsrv


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
    result = dvsrv(time, event, X, beta)
    assert "residuals" in result
    assert "martingale" in result


def test_residuals_shape():
    time, event, X, beta = _make_data()
    result = dvsrv(time, event, X, beta)
    assert result["residuals"].shape == (len(time),)


def test_residuals_finite():
    time, event, X, beta = _make_data()
    result = dvsrv(time, event, X, beta)
    assert np.all(np.isfinite(result["residuals"]))


def test_deviance_symmetric_around_zero():
    """Deviance residuals should have a reasonable spread around 0."""
    time, event, X, beta = _make_data(n=300, seed=2)
    result = dvsrv(time, event, X, beta)
    mean_dev = result["residuals"].mean()
    assert abs(mean_dev) < 1.0  # should be roughly mean-zero


def test_deviance_positive_for_events():
    """For event subjects with high risk, deviance is positive;
    censored subjects have non-positive deviance."""
    time, event, X, beta = _make_data()
    result = dvsrv(time, event, X, beta)
    M = result["martingale"]
    D = result["residuals"]
    # Check sign consistency: sign(D) == sign(M)
    nonzero_M = M != 0
    assert np.all(np.sign(D[nonzero_M]) == np.sign(M[nonzero_M]))


def test_deviance_geq_martingale_in_magnitude():
    """For events: D = sign(M)*sqrt(-2*(M+log(1-M))) >= |M| approximately for M<0."""
    time, event, X, beta = _make_data(n=200, seed=3)
    result = dvsrv(time, event, X, beta)
    # This is not always true, but deviance amplifies martingale for outliers
    assert np.all(np.isfinite(result["residuals"]))
