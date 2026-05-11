"""Tests for loocv.py - Leave-One-Out Cross-Validation."""
import numpy as np
import pytest
from morie.fn.loocv import loocv_fn, loocv_fn_alias


def _two_class_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((10, 2)) + np.array([0, 0])
    X1 = rng.standard_normal((10, 2)) + np.array([5, 5])
    X = np.vstack([X0, X1])
    y = np.array([0]*10 + [1]*10)
    return X, y


def test_loocv_returns_descriptive_result():
    X, y = _two_class_data()
    result = loocv_fn(X, y)
    assert result.name == "loocv"
    assert "accuracy" in result.extra


def test_loocv_well_separated():
    X, y = _two_class_data()
    result = loocv_fn(X, y)
    assert result.value > 0.7


def test_loocv_bounded():
    X, y = _two_class_data()
    result = loocv_fn(X, y)
    assert 0.0 <= result.value <= 1.0


def test_loocv_alias():
    X, y = _two_class_data()
    result = loocv_fn_alias(X, y)
    assert result.name == "loocv"
