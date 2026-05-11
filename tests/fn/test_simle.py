"""Tests for morie.fn.simle — Single-index MLE."""

import numpy as np
import pytest
from morie.fn.simle import simle


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    prob = 1 / (1 + np.exp(-(X @ np.array([1, 0.5]))))
    y = (rng.uniform(size=n) < prob).astype(float)
    result = simle(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "index", "g_hat", "log_likelihood", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = simle(y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_non_binary_raises():
    with pytest.raises(ValueError, match="binary"):
        simle(np.array([0, 1, 2, 0, 1]), np.ones((5, 2)))


def test_probabilities_bounded():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = simle(y, X)
    assert all(0 <= p <= 1 for p in result["g_hat"])
