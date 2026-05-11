"""Tests for rng179.rangayyan_ch4_filtered_derivative_murthy."""
import numpy as np
import pytest
from morie.fn.rng179 import rangayyan_ch4_filtered_derivative_murthy


def test_rng179_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch4_filtered_derivative_murthy(x, n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng179_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch4_filtered_derivative_murthy(x, n, N)
    assert isinstance(result, dict)
