"""Tests for bkprd.py - Backward prediction error."""
import numpy as np
from moirais.fn.bkprd import backward_prediction_fn, bkprd


def test_bkprd_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    ar = np.array([0.5, -0.3])
    result = backward_prediction_fn(x, ar)
    assert result.name == "backward_prediction"
    assert "error" in result.extra
    assert result.extra["mse"] >= 0


def test_bkprd_alias():
    x = np.random.default_rng(42).standard_normal(64)
    ar = np.array([0.3])
    result = bkprd(x, ar)
    assert result.name == "backward_prediction"
