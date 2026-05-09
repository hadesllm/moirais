"""Tests for moirais.fn.plrob — Robinson partially linear estimator."""

import numpy as np
import pytest
from moirais.fn.plrob import plrob


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 2))
    y = 2 * X[:, 0] - X[:, 1] + np.sin(2 * np.pi * Z) + rng.normal(0, 0.3, n)
    result = plrob(y, X, Z)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "residuals", "n_obs"):
        assert key in result


def test_beta_sign():
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = 3.0 * X[:, 0] + np.sin(Z) + rng.normal(0, 0.2, n)
    result = plrob(y, X, Z)
    assert result["beta"][0] > 0


def test_se_positive():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + Z + rng.normal(0, 0.1, n)
    result = plrob(y, X, Z)
    assert all(s > 0 for s in result["se"])


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 10"):
        plrob(np.ones(5), np.ones((5, 1)), np.ones(5))
