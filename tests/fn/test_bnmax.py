"""Tests for moirais.fn.bnmax — Maximum score estimator."""

import numpy as np
import pytest
from moirais.fn.bnmax import bnmax


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    result = bnmax(y, X, seed=42)
    assert isinstance(result, dict)
    for key in ("beta", "score", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnmax(y, X, seed=42)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_score_bounded():
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnmax(y, X, seed=42)
    assert -1 <= result["score"] <= 1


def test_non_binary_raises():
    with pytest.raises(ValueError, match="binary"):
        bnmax(np.array([0, 1, 2, 0, 1]), np.ones((5, 2)))
