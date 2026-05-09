"""Tests for arprd.py - AR model prediction."""
import numpy as np
import pytest
from moirais.fn.arprd import ar_predict_fn, arprd


def test_ar_predict_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_predict_fn(x)
    assert result.name == "ar_predict"
    assert "predictions" in result.extra


def test_ar_predict_single_step():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_predict_fn(x, n_ahead=1)
    assert len(result.extra["predictions"]) == 1
    assert result.value is not None


def test_ar_predict_multi_step():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_predict_fn(x, n_ahead=5)
    assert len(result.extra["predictions"]) == 5


def test_arprd_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = arprd(x, order=5, n_ahead=3)
    assert result.name == "ar_predict"
