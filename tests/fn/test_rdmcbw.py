"""Tests for rdmcbw.mse_optimal_bandwidth_rdd."""
import numpy as np
import pytest
from morie.fn.rdmcbw import mse_optimal_bandwidth_rdd


def test_rdmcbw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = mse_optimal_bandwidth_rdd(y, x, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdmcbw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = mse_optimal_bandwidth_rdd(y, x, cutoff)
    assert isinstance(result, dict)
