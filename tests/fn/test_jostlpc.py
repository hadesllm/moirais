"""Tests for jostlpc.joseph_stl_decomposition."""
import numpy as np
import pytest
from moirais.fn.jostlpc import joseph_stl_decomposition


def test_jostlpc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    seasonal_window = np.random.default_rng(42).normal(0, 1, 100)
    trend_window = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_stl_decomposition(y, period, seasonal_window, trend_window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jostlpc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    seasonal_window = np.random.default_rng(42).normal(0, 1, 100)
    trend_window = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_stl_decomposition(y, period, seasonal_window, trend_window)
    assert isinstance(result, dict)
