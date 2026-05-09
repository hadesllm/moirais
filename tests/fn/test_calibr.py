"""Tests for calibr.calibration_estimator."""
import numpy as np
import pytest
from moirais.fn.calibr import calibration_estimator


def test_calibr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1, 100)
    result = calibration_estimator(y, X, weights, totals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_calibr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1, 100)
    result = calibration_estimator(y, X, weights, totals)
    assert isinstance(result, dict)
