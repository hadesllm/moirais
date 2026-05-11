"""Tests for morie.fn.oracl — oracle inequality verification."""

import numpy as np
import pytest

from morie.fn.oracl import oracl


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + rng.standard_normal(n) * 0.5
    result = oracl(X, y, n_folds=3, seed=7)
    assert "selected_idx" in result
    assert "cv_risks" in result
    assert "satisfies_oracle" in result


def test_ratio_ge_one():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n)
    result = oracl(X, y, n_folds=3, seed=42)
    assert result["ratio"] >= 1.0 - 1e-6


def test_satisfies_oracle():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n)
    result = oracl(X, y, n_folds=5, seed=1)
    assert result["satisfies_oracle"] is True


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        oracl(np.array([]).reshape(0, 1), np.array([]))
