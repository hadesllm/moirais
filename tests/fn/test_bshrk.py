"""Tests for moirais.fn.bshrk -- Bayesian horseshoe."""

import numpy as np
from moirais.fn.bshrk import bayesian_horseshoe


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = X[:, 0] * 3 + rng.standard_normal(50) * 0.5
    result = bayesian_horseshoe(X, y, n_iter=500)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_shrinks_zero_coefs():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] * 3 + rng.standard_normal(100) * 0.5
    result = bayesian_horseshoe(X, y, n_iter=1000)
    assert abs(result["posterior_mean"][0]) > abs(result["posterior_mean"][4])


def test_samples_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    result = bayesian_horseshoe(X, y, n_iter=200)
    assert result["beta_samples"].shape == (200, 3)
