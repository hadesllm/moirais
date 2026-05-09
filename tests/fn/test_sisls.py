"""Tests for moirais.fn.sisls — Single-index via iterative SLS."""

import numpy as np
import pytest
from moirais.fn.sisls import sisls


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3))
    y = (X @ np.array([1, 0.5, 0]))**2 + rng.normal(0, 0.3, n)
    result = sisls(y, X, max_iter=10)
    assert isinstance(result, dict)
    for key in ("beta", "index", "g_hat", "n_iter", "converged", "n_obs"):
        assert key in result


def test_beta_unit_norm():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = np.sin(X @ np.array([1, 0])) + rng.normal(0, 0.1, n)
    result = sisls(y, X, max_iter=5)
    beta = np.asarray(result["beta"])
    assert abs(np.linalg.norm(beta) - 1.0) < 1e-4


def test_too_few_covariates():
    with pytest.raises(ValueError, match="p >= 2"):
        sisls(np.ones(20), np.ones(20))


def test_n_iter_positive():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    y = X[:, 0] + rng.normal(0, 0.1, 50)
    result = sisls(y, X, max_iter=3)
    assert result["n_iter"] >= 1
