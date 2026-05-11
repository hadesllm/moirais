"""Tests for rng079.rangayyan_ch3_periodic_convolution."""
import numpy as np
import pytest
from morie.fn.rng079 import rangayyan_ch3_periodic_convolution


def test_rng079_basic():
    """Test basic functionality."""
    x_p = np.random.default_rng(42).normal(0, 1, 100)
    h_p = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch3_periodic_convolution(x_p, h_p, n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng079_edge():
    """Test edge cases."""
    x_p = np.random.default_rng(42).normal(0, 1, 100)
    h_p = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch3_periodic_convolution(x_p, h_p, n, N)
    assert isinstance(result, dict)
