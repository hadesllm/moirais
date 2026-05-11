"""Tests for cnnbs.py - 1D CNN for biosignals."""
import numpy as np
import pytest
from morie.fn.cnnbs import cnnbs_fn, cnnbs


def test_cnnbs_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 20))
    y = np.array([0.0]*20 + [1.0]*20)
    result = cnnbs_fn(X, y, n_epochs=3, n_filters=4, kernel_size=3)
    assert result.name == "cnn_biosignal"
    assert "filters" in result.extra
    assert "accuracy" in result.extra


def test_cnnbs_accuracy_bounded():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 20))
    y = np.array([0.0]*20 + [1.0]*20)
    result = cnnbs_fn(X, y, n_epochs=3)
    assert 0.0 <= result.value <= 1.0


def test_cnnbs_filter_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 16))
    y = np.array([0.0]*15 + [1.0]*15)
    result = cnnbs_fn(X, y, n_filters=4, kernel_size=3, n_epochs=2)
    assert result.extra["filters"].shape == (4, 3)


def test_cnnbs_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 12))
    y = np.array([0.0]*10 + [1.0]*10)
    result = cnnbs(X, y, n_epochs=2, n_filters=2, kernel_size=3)
    assert result.name == "cnn_biosignal"
