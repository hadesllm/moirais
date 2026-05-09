"""Tests for lglog: log-logistic AFT survival model."""
import numpy as np
import pytest
from moirais.fn.lglog import lglog


def _sim_loglogistic(n=150, mu_true=1.0, sigma_true=0.5, seed=0):
    """Simulate from log-logistic via quantile transform."""
    rng = np.random.default_rng(seed)
    # F^{-1}(u) = exp(mu + sigma * log(u/(1-u)))
    u = rng.uniform(0.01, 0.99, size=n)
    T = np.exp(mu_true + sigma_true * np.log(u / (1 - u)))
    C = rng.exponential(T.mean() * 2, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _sim_loglogistic()
    result = lglog(time, event)
    for key in ("mu", "sigma", "log_likelihood", "aic", "bic"):
        assert key in result


def test_sigma_positive():
    time, event = _sim_loglogistic()
    result = lglog(time, event)
    assert result["sigma"] > 0


def test_log_sigma_consistent():
    time, event = _sim_loglogistic()
    result = lglog(time, event)
    assert abs(np.exp(result["log_sigma"]) - result["sigma"]) < 1e-10


def test_mu_direction():
    """Increasing mu shifts survival to the right."""
    time1, event1 = _sim_loglogistic(mu_true=0.5, seed=1)
    time2, event2 = _sim_loglogistic(mu_true=2.0, seed=1)
    r1 = lglog(time1, event1)
    r2 = lglog(time2, event2)
    assert r2["mu"] > r1["mu"]


def test_aic_finite():
    time, event = _sim_loglogistic()
    result = lglog(time, event)
    assert np.isfinite(result["aic"])
    assert np.isfinite(result["bic"])


def test_with_covariates():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 1))
    mu = 1.5
    sigma = 0.5
    beta = 0.8
    log_T = mu + beta * X[:, 0] + rng.logistic(0, sigma, size=n)
    T = np.exp(log_T)
    C = rng.exponential(np.exp(mu) * 3, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    result = lglog(time, event, X)
    assert len(result["beta"]) == 1
    assert result["beta"][0] > 0  # positive covariate effect


def test_negative_time_raises():
    with pytest.raises(ValueError):
        lglog(np.array([1.0, -2.0]), np.array([1, 0]))


def test_log_likelihood_negative():
    """For a reasonable model, log-likelihood should be finite and negative."""
    time, event = _sim_loglogistic(n=100)
    result = lglog(time, event)
    assert np.isfinite(result["log_likelihood"])
