"""Tests for fingy: Fine-Gray subdistribution hazard model."""
import numpy as np
import pytest
from moirais.fn.fingy import fingy


def _sim_competing_risks_cov(n=200, beta_true=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    # Cause-1 subdistribution hazard depends on X
    T1 = rng.exponential(1.0 / np.exp(beta_true * X[:, 0]))
    T2 = rng.exponential(2.0, size=n)
    C = rng.exponential(5.0, size=n)
    time = np.minimum(np.minimum(T1, T2), C)
    event = np.where(T1 < T2, 1, 2)
    event = np.where(C < np.minimum(T1, T2), 0, event)
    return time, event.astype(int), X


def test_returns_keys():
    time, event, X = _sim_competing_risks_cov()
    result = fingy(time, event, X, cause=1)
    for key in ("beta", "se", "z", "p_value", "log_likelihood", "converged"):
        assert key in result


def test_beta_shape():
    time, event, X = _sim_competing_risks_cov()
    result = fingy(time, event, X, cause=1)
    assert result["beta"].shape == (1,)


def test_beta_positive_when_true_positive():
    """Positive subdistribution hazard effect should give positive beta."""
    time, event, X = _sim_competing_risks_cov(n=300, beta_true=0.8, seed=1)
    result = fingy(time, event, X, cause=1)
    assert result["beta"][0] > 0


def test_se_positive():
    time, event, X = _sim_competing_risks_cov()
    result = fingy(time, event, X, cause=1)
    assert np.all(result["se"] > 0)


def test_p_value_range():
    time, event, X = _sim_competing_risks_cov()
    result = fingy(time, event, X, cause=1)
    assert np.all(result["p_value"] >= 0)
    assert np.all(result["p_value"] <= 1)


def test_cause_not_found_raises():
    time, event, X = _sim_competing_risks_cov()
    with pytest.raises(ValueError):
        fingy(time, event, X, cause=99)


def test_negative_time_raises():
    time = np.array([-1.0, 2.0, 3.0])
    event = np.array([1, 2, 0])
    X = np.ones((3, 1))
    with pytest.raises(ValueError):
        fingy(time, event, X)


def test_log_likelihood_finite():
    time, event, X = _sim_competing_risks_cov()
    result = fingy(time, event, X, cause=1)
    assert np.isfinite(result["log_likelihood"])
