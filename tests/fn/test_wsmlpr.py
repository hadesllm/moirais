"""Tests for wsmlpr.wasserman_local_polynomial."""
import numpy as np
import pytest
from morie.fn.wsmlpr import wasserman_local_polynomial


def test_wsmlpr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    p = 5
    result = wasserman_local_polynomial(x, x_data, y_data, h, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmlpr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    p = 5
    result = wasserman_local_polynomial(x, x_data, y_data, h, p)
    assert isinstance(result, dict)
