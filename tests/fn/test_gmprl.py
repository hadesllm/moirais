"""Tests for gmprl: Gompertz survival model."""
import numpy as np
import pytest
from moirais.fn.gmprl import gmprl


def _sim_gompertz(n=200, log_lam=-1.0, gamma=0.3, seed=0):
    """Simulate Gompertz survival times via inverse CDF."""
    rng = np.random.default_rng(seed)
    lam = np.exp(log_lam)
    u = rng.uniform(0.001, 0.999, size=n)
    if abs(gamma) < 1e-10:
        T = -np.log(u) / lam
    else:
        T = np.log(1 - gamma / lam * np.log(u)) / gamma
    T = np.maximum(T, 1e-6)
    C = rng.exponential(T.mean() * 2, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _sim_gompertz()
    result = gmprl(time, event)
    for key in ("log_lambda", "gamma", "beta", "log_likelihood", "aic", "bic"):
        assert key in result


def test_gamma_direction():
    """Positive gamma_true should give positive gamma estimate."""
    time, event = _sim_gompertz(gamma=0.5, seed=1)
    result = gmprl(time, event)
    assert result["gamma"] > 0


def test_gamma_near_zero_exponential():
    """gamma=0 (exponential) should give near-zero gamma estimate."""
    time, event = _sim_gompertz(n=300, gamma=0.0, seed=2)
    result = gmprl(time, event)
    assert abs(result["gamma"]) < 0.5


def test_aic_formula():
    time, event = _sim_gompertz()
    result = gmprl(time, event)
    n = len(time)
    k = 2  # log_lambda + gamma, no covariates
    expected_aic = -2 * result["log_likelihood"] + 2 * k
    assert abs(result["aic"] - expected_aic) < 1e-6


def test_with_covariates():
    rng = np.random.default_rng(5)
    n = 200
    X = rng.standard_normal((n, 1))
    log_lam = -1.0
    gamma = 0.2
    beta_true = 0.5
    lam = np.exp(log_lam) * np.exp(beta_true * X[:, 0])
    u = rng.uniform(0.001, 0.999, size=n)
    T = np.log(1 - gamma / lam * np.log(u)) / gamma
    T = np.maximum(T, 1e-6)
    C = rng.exponential(T.mean() * 2, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    result = gmprl(time, event, X)
    assert len(result["beta"]) == 1
    # Positive covariate should increase hazard => positive beta
    assert result["beta"][0] > 0


def test_negative_time_raises():
    with pytest.raises(ValueError):
        gmprl(np.array([1.0, -1.0, 2.0]), np.array([1, 0, 1]))


def test_log_likelihood_finite():
    time, event = _sim_gompertz()
    result = gmprl(time, event)
    assert np.isfinite(result["log_likelihood"])
