"""Tests for rng122.rangayyan_ch3_baseline_wander_filter_difference_eq."""
import numpy as np
import pytest
from moirais.fn.rng122 import rangayyan_ch3_baseline_wander_filter_difference_eq


def test_rng122_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_baseline_wander_filter_difference_eq(x, y, T, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng122_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_baseline_wander_filter_difference_eq(x, y, T, n)
    assert isinstance(result, dict)
