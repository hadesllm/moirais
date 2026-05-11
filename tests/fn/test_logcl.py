"""Tests for logcl.py - Logistic classifier."""
import numpy as np
import pytest
from morie.fn.logcl import logcl_fn, logcl


def _separable_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((50, 2)) + np.array([-3, -3])
    X1 = rng.standard_normal((50, 2)) + np.array([3, 3])
    X = np.vstack([X0, X1])
    y = np.array([0.0]*50 + [1.0]*50)
    return X, y


def test_logcl_returns_descriptive_result():
    X, y = _separable_data()
    result = logcl_fn(X, y)
    assert result.name == "logistic"
    assert "weights" in result.extra
    assert "accuracy" in result.extra


def test_logcl_high_accuracy():
    X, y = _separable_data()
    result = logcl_fn(X, y, n_iter=2000)
    assert result.value > 0.8


def test_logcl_weight_shape():
    X, y = _separable_data()
    result = logcl_fn(X, y)
    assert len(result.extra["weights"]) == X.shape[1] + 1


def test_logcl_alias():
    X, y = _separable_data()
    result = logcl(X, y, n_iter=100)
    assert result.name == "logistic"
