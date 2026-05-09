"""Tests for mrsrv: martingale residuals for Cox model."""
import numpy as np
import pytest
from moirais.fn.mrsrv import mrsrv


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
    result = mrsrv(time, event, X, beta)
    for key in ("residuals", "baseline_cumhaz", "expected"):
        assert key in result


def test_residuals_shape():
    time, event, X, beta = _make_data()
    result = mrsrv(time, event, X, beta)
    assert result["residuals"].shape == (len(time),)


def test_residuals_sum_near_zero():
    """Martingale residuals sum to (approximately) zero for a well-fitted model."""
    time, event, X, beta = _make_data(n=300, seed=1)
    result = mrsrv(time, event, X, beta)
    # Not exact but should be small relative to n
    assert abs(result["residuals"].sum()) < 20


def test_residuals_leq_1():
    """M_i = delta_i - E_i <= 1 always."""
    time, event, X, beta = _make_data()
    result = mrsrv(time, event, X, beta)
    assert np.all(result["residuals"] <= 1.0 + 1e-12)


def test_expected_nonneg():
    time, event, X, beta = _make_data()
    result = mrsrv(time, event, X, beta)
    assert np.all(result["expected"] >= 0)


def test_baseline_cumhaz_monotone():
    time, event, X, beta = _make_data()
    result = mrsrv(time, event, X, beta)
    order = np.argsort(time)
    H = result["baseline_cumhaz"][order]
    assert np.all(np.diff(H) >= -1e-12)


def test_censored_residuals_leq_zero():
    """Censored martingale residuals M_i = 0 - H_0(t_i)*exp(eta_i) <= 0."""
    time, event, X, beta = _make_data()
    result = mrsrv(time, event, X, beta)
    censored = event == 0
    assert np.all(result["residuals"][censored] <= 0.0 + 1e-12)


def test_negative_time_raises():
    with pytest.raises(ValueError):
        mrsrv(np.array([-1.0, 2.0]), np.array([1, 0]),
              np.array([[1.0], [0.0]]), np.array([0.5]))
