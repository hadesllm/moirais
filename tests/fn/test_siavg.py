"""Tests for morie.fn.siavg — Single-index via average derivative."""

import numpy as np
import pytest
from morie.fn.siavg import siavg


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    beta_true = np.array([1, 0.5, -0.5])
    y = np.sin(X @ beta_true) + rng.normal(0, 0.1, n)
    result = siavg(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "index", "avg_derivative", "se", "n_obs", "p"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X @ np.array([1, 0, 0]) + rng.normal(0, 0.1, n)
    result = siavg(y, X)
    beta = np.asarray(result["beta"])
    assert abs(np.linalg.norm(beta) - 1.0) < 1e-6


def test_too_few_covariates_raises():
    with pytest.raises(ValueError, match="p >= 2"):
        siavg(np.ones(20), np.ones(20))


def test_mismatch_raises():
    with pytest.raises(ValueError, match="!="):
        siavg(np.ones(10), np.ones((5, 2)))
