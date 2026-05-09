"""Tests for moirais.fn.bnsmo — Smoothed maximum score."""

import numpy as np
import pytest
from moirais.fn.bnsmo import bnsmo


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    result = bnsmo(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "score", "bandwidth", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnsmo(y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_non_binary_raises():
    with pytest.raises(ValueError, match="binary"):
        bnsmo(np.array([0, 1, 2] * 5), np.ones((15, 2)))


def test_bandwidth_positive():
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnsmo(y, X)
    assert result["bandwidth"] > 0
