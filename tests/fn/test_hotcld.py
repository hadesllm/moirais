"""Tests for hotcld.hot_cold_spots."""
import numpy as np
import pytest
from morie.fn.hotcld import hot_cold_spots


def test_hotcld_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = hot_cold_spots(x, W, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hotcld_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = hot_cold_spots(x, W, alpha)
    assert isinstance(result, dict)
