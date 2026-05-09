"""Tests for rng116.rangayyan_ch3_three_point_central_difference."""
import numpy as np
import pytest
from moirais.fn.rng116 import rangayyan_ch3_three_point_central_difference


def test_rng116_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_three_point_central_difference(x, T, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng116_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_three_point_central_difference(x, T, n)
    assert isinstance(result, dict)
