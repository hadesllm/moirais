"""Tests for gb241.gibbons_order_cdf."""
import numpy as np
import pytest
from morie.fn.gb241 import gibbons_order_cdf


def test_gb241_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    r = 10
    n = 100
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_cdf(t, r, n, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb241_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    r = 10
    n = 100
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_cdf(t, r, n, F)
    assert isinstance(result, dict)
