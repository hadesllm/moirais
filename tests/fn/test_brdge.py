"""Tests for moirais.fn.brdge -- Bayesian ridge regression."""

import numpy as np
from moirais.fn.brdge import bayesian_ridge


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X @ [1, 2, 0] + rng.standard_normal(50) * 0.5
    result = bayesian_ridge(X, y)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_alpha_positive():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X @ [1, 2, 0] + rng.standard_normal(50)
    result = bayesian_ridge(X, y)
    assert result["alpha"] > 0


def test_ci_contains_mean():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = X @ [3, -1] + rng.standard_normal(100) * 0.3
    result = bayesian_ridge(X, y)
    for j in range(2):
        assert result["ci_lower"][j] < result["posterior_mean"][j] < result["ci_upper"][j]
