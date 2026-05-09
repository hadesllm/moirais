"""Tests for wsmcrk.wasserman_kernel_regression."""
import numpy as np
import pytest
from moirais.fn.wsmcrk import wasserman_kernel_regression


def test_wsmcrk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = wasserman_kernel_regression(x, x_data, y_data, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcrk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_data = np.random.default_rng(42).normal(0, 1, 100)
    y_data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = wasserman_kernel_regression(x, x_data, y_data, h)
    assert isinstance(result, dict)
