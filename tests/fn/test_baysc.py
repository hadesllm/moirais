"""Tests for baysc.py - Bayesian classifier."""
import numpy as np
import pytest
from moirais.fn.baysc import baysc_fn, baysc


def _two_class_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((30, 2)) + np.array([0, 0])
    X1 = rng.standard_normal((30, 2)) + np.array([5, 5])
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    return X, y


def test_baysc_returns_descriptive_result():
    X, y = _two_class_data()
    result = baysc_fn(X, y, X[:5])
    assert result.name == "bayes_classifier"
    assert "predictions" in result.extra
    assert "posteriors" in result.extra


def test_baysc_well_separated():
    X, y = _two_class_data()
    result = baysc_fn(X, y, X)
    preds = result.extra["predictions"]
    accuracy = np.mean(preds == y)
    assert accuracy > 0.8


def test_baysc_prediction_count():
    X, y = _two_class_data()
    X_test = X[:10]
    result = baysc_fn(X, y, X_test)
    assert len(result.extra["predictions"]) == 10


def test_baysc_alias():
    X, y = _two_class_data()
    result = baysc(X, y, X[:3])
    assert result.name == "bayes_classifier"
