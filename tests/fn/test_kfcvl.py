"""Tests for kfcvl.py - K-fold cross-validation."""
import numpy as np
import pytest
from moirais.fn.kfcvl import kfcvl_fn, kfcvl


def _two_class_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((30, 2)) + np.array([0, 0])
    X1 = rng.standard_normal((30, 2)) + np.array([5, 5])
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    return X, y


def test_kfcvl_returns_descriptive_result():
    X, y = _two_class_data()
    result = kfcvl_fn(X, y, k=3)
    assert result.name == "kfold_cv"
    assert "mean_accuracy" in result.extra
    assert "fold_accuracies" in result.extra
    assert result.extra["k"] == 3


def test_kfcvl_fold_count():
    X, y = _two_class_data()
    result = kfcvl_fn(X, y, k=5)
    assert len(result.extra["fold_accuracies"]) == 5


def test_kfcvl_reasonable_accuracy():
    X, y = _two_class_data()
    result = kfcvl_fn(X, y, k=3)
    assert result.value > 0.5


def test_kfcvl_alias():
    X, y = _two_class_data()
    result = kfcvl(X, y, k=2)
    assert result.name == "kfold_cv"
