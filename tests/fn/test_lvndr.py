"""Tests for lvndr.py - Levinson-Durbin recursion."""
import numpy as np
import pytest
from moirais.fn.lvndr import levinson_durbin_fn, lvndr


def test_levinson_durbin_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = levinson_durbin_fn(x)
    assert result.name == "levinson_durbin"
    assert "coefficients" in result.extra
    assert "prediction_error" in result.extra


def test_levinson_durbin_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = levinson_durbin_fn(x, order=8)
    assert len(result.extra["coefficients"]) == 8


def test_levinson_durbin_prediction_error_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = levinson_durbin_fn(x)
    assert result.extra["prediction_error"] > 0


def test_lvndr_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = lvndr(x, order=5)
    assert result.name == "levinson_durbin"
