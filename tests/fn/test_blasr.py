"""Tests for morie.fn.blasr -- Bayesian LASSO."""

import numpy as np
from morie.fn.blasr import bayesian_lasso


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X[:, 0] * 2 + rng.standard_normal(50)
    result = bayesian_lasso(X, y, n_iter=500)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_median_exists():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X[:, 0] * 2 + rng.standard_normal(50)
    result = bayesian_lasso(X, y, n_iter=500)
    assert len(result["posterior_median"]) == 3


def test_beta_samples_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 2))
    y = rng.standard_normal(30)
    result = bayesian_lasso(X, y, n_iter=200)
    assert result["beta_samples"].shape == (200, 2)
