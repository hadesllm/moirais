"""Tests for prder.py - Forward prediction error."""
import numpy as np
from morie.fn.prder import prediction_error_fn, prder


def test_prder_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    ar = np.array([0.5, -0.3])
    result = prediction_error_fn(x, ar)
    assert result.name == "prediction_error"
    assert "error" in result.extra
    assert result.extra["mse"] >= 0


def test_prder_error_length():
    x = np.random.default_rng(42).standard_normal(100)
    ar = np.array([0.5])
    result = prediction_error_fn(x, ar)
    assert len(result.extra["error"]) == 100


def test_prder_alias():
    x = np.random.default_rng(42).standard_normal(64)
    ar = np.array([0.3])
    result = prder(x, ar)
    assert result.name == "prediction_error"
