"""Tests for morie.fn.simin — Single-index minimum distance."""

import numpy as np
import pytest
from morie.fn.simin import simin


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = np.sin(X @ np.array([1, 0.5])) + rng.normal(0, 0.1, n)
    result = simin(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "index", "g_hat", "min_distance", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 2))
    y = X[:, 0] + rng.normal(0, 0.1, 80)
    result = simin(y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_min_distance_nonneg():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 2))
    y = X[:, 0] + rng.normal(0, 0.1, 60)
    result = simin(y, X)
    assert result["min_distance"] >= 0
