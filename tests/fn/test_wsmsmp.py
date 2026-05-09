"""Tests for wsmsmp.wasserman_smoothing_spline."""
import numpy as np
import pytest
from moirais.fn.wsmsmp import wasserman_smoothing_spline


def test_wsmsmp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_smoothing_spline(x, y, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmsmp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_smoothing_spline(x, y, lambda_)
    assert isinstance(result, dict)
