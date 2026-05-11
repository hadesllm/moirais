"""Tests for flda.py - Fisher Linear Discriminant Analysis."""
import numpy as np
import pytest
from morie.fn.flda import flda_fn, flda


def _two_class_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((30, 3)) + np.array([0, 0, 0])
    X1 = rng.standard_normal((30, 3)) + np.array([3, 3, 3])
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    return X, y


def test_flda_returns_descriptive_result():
    X, y = _two_class_data()
    result = flda_fn(X, y)
    assert result.name == "fisher_lda"
    assert "weights" in result.extra
    assert "means" in result.extra
    assert "threshold" in result.extra


def test_flda_weight_unit_norm():
    X, y = _two_class_data()
    result = flda_fn(X, y)
    w = result.extra["weights"]
    assert abs(np.linalg.norm(w) - 1.0) < 1e-6


def test_flda_separates_classes():
    X, y = _two_class_data()
    result = flda_fn(X, y)
    w = result.extra["weights"]
    proj = X @ w
    assert np.mean(proj[y == 0]) < np.mean(proj[y == 1])


def test_flda_alias():
    X, y = _two_class_data()
    result = flda(X, y)
    assert result.name == "fisher_lda"
