"""Tests for morie.fn.siprj — Single-index projection pursuit."""

import numpy as np
import pytest
from morie.fn.siprj import siprj


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3))
    y = np.sin(X @ np.array([1, 0, 0])) + rng.normal(0, 0.1, n)
    result = siprj(y, X, max_iter=5)
    assert isinstance(result, dict)
    for key in ("beta", "index", "g_hat", "rss", "n_iter", "converged", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 2))
    y = X[:, 0]**2 + rng.normal(0, 0.1, 80)
    result = siprj(y, X, max_iter=5)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_rss_nonneg():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 2))
    y = X[:, 0] + rng.normal(0, 0.1, 60)
    result = siprj(y, X, max_iter=3)
    assert result["rss"] >= 0
