"""Tests for qda.py - Quadratic Discriminant Analysis."""
import numpy as np
import pytest
from moirais.fn.qda import qda_fn, qda


def _two_class_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((30, 2)) + np.array([0, 0])
    X1 = rng.standard_normal((30, 2)) + np.array([5, 5])
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    return X, y


def test_qda_returns_descriptive_result():
    X, y = _two_class_data()
    result = qda_fn(X, y, X[:5])
    assert result.name == "qda"
    assert "predictions" in result.extra


def test_qda_well_separated():
    X, y = _two_class_data()
    result = qda_fn(X, y, X)
    preds = result.extra["predictions"]
    accuracy = np.mean(preds == y)
    assert accuracy > 0.8


def test_qda_prediction_count():
    X, y = _two_class_data()
    result = qda_fn(X, y, X[:10])
    assert result.value == 10


def test_qda_alias():
    X, y = _two_class_data()
    result = qda(X, y, X[:3])
    assert result.name == "qda"
