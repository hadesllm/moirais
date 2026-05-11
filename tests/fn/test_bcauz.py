"""Tests for morie.fn.bcauz -- Bayesian ATE."""

import numpy as np
from morie.fn.bcauz import bayesian_ate


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    t = np.concatenate([np.zeros(50), np.ones(50)])
    y = t * 2 + rng.standard_normal(n) * 0.5
    result = bayesian_ate(y, t, n_iter=1000)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_ate_near_truth():
    rng = np.random.default_rng(42)
    t = np.concatenate([np.zeros(100), np.ones(100)])
    y = t * 3 + rng.standard_normal(200)
    result = bayesian_ate(y, t, n_iter=2000)
    assert abs(result["posterior_mean"] - 3.0) < 1.5


def test_hdi_contains_mean():
    rng = np.random.default_rng(42)
    t = np.concatenate([np.zeros(50), np.ones(50)])
    y = t * 2 + rng.standard_normal(100)
    result = bayesian_ate(y, t, n_iter=1000)
    assert result["hdi_lower"] < result["posterior_mean"] < result["hdi_upper"]
