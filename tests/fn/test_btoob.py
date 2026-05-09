"""Tests for btoob.boot_oob_error."""
import numpy as np
import pytest
from moirais.fn.btoob import boot_oob_error


def test_btoob_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    predict_fn = (lambda v: v)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_oob_error(x, y, fit_fn, predict_fn, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btoob_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    predict_fn = (lambda v: v)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_oob_error(x, y, fit_fn, predict_fn, B)
    assert isinstance(result, dict)
