"""Tests for moirais.fn.bivrt -- Bayesian IV regression."""

import numpy as np
from moirais.fn.bivrt import bayesian_iv


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.standard_normal((n, 1))
    X = Z + rng.standard_normal((n, 1)) * 0.5
    y = X.ravel() * 2 + rng.standard_normal(n)
    result = bayesian_iv(y, X, Z, n_iter=500)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_beta_near_truth():
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.standard_normal((n, 1))
    X = Z * 2 + rng.standard_normal((n, 1)) * 0.3
    y = X.ravel() * 3 + rng.standard_normal(n) * 0.5
    result = bayesian_iv(y, X, Z, n_iter=1000)
    assert abs(result["posterior_mean"][0] - 3.0) < 2.0


def test_ci_lower_upper():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.standard_normal((n, 1))
    X = Z + rng.standard_normal((n, 1))
    y = X.ravel() * 2 + rng.standard_normal(n)
    result = bayesian_iv(y, X, Z, n_iter=500)
    assert result["ci_lower"][0] < result["ci_upper"][0]
