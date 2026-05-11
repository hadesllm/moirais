"""Tests for lgnrm: log-normal AFT survival model."""
import numpy as np
import pytest
from morie.fn.lgnrm import lgnrm


def _sim_lognormal(n=150, mu_true=1.5, sigma_true=0.5, censor_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    T = np.exp(rng.normal(mu_true, sigma_true, size=n))
    C = rng.exponential(T.mean() / censor_rate, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_intercept_only_returns_keys():
    time, event = _sim_lognormal()
    result = lgnrm(time, event)
    for key in ("mu", "sigma", "log_likelihood", "aic", "bic"):
        assert key in result


def test_mu_estimate_close_to_truth():
    time, event = _sim_lognormal(n=500, mu_true=2.0, sigma_true=0.4, seed=1)
    result = lgnrm(time, event)
    assert abs(result["mu"] - 2.0) < 0.3


def test_sigma_positive():
    time, event = _sim_lognormal()
    result = lgnrm(time, event)
    assert result["sigma"] > 0


def test_log_sigma_consistent():
    time, event = _sim_lognormal()
    result = lgnrm(time, event)
    assert abs(np.log(result["sigma"]) - result["log_sigma"]) < 1e-10


def test_aic_gt_neg2ll():
    time, event = _sim_lognormal()
    result = lgnrm(time, event)
    # AIC = -2*ll + 2*k, k >= 2
    assert result["aic"] > -2 * result["log_likelihood"]


def test_bic_formula():
    n = 150
    time, event = _sim_lognormal(n=n)
    result = lgnrm(time, event)
    # BIC = -2*ll + k*log(n), k=2 for intercept-only
    expected_bic = -2 * result["log_likelihood"] + 2 * np.log(n)
    assert abs(result["bic"] - expected_bic) < 1e-6


def test_with_covariates():
    rng = np.random.default_rng(5)
    n = 200
    X = rng.standard_normal((n, 2))
    beta_true = np.array([0.5, -0.3])
    mu = 2.0
    sigma = 0.6
    log_T = mu + X @ beta_true + rng.normal(0, sigma, size=n)
    T = np.exp(log_T)
    C = rng.exponential(T.mean() * 2, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    result = lgnrm(time, event, X)
    assert len(result["beta"]) == 2
    # Beta signs should match
    assert result["beta"][0] > 0
    assert result["beta"][1] < 0


def test_negative_time_raises():
    time = np.array([1.0, -2.0, 3.0])
    event = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        lgnrm(time, event)


def test_non_binary_event_raises():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 2, 0])
    with pytest.raises(ValueError):
        lgnrm(time, event)
