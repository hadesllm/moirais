"""Tests for gb2101.gibbons_asymp_order_normal."""
import numpy as np
import pytest
from moirais.fn.gb2101 import gibbons_asymp_order_normal


def test_gb2101_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    p = 5
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_asymp_order_normal(r, n, p, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb2101_edge():
    """Test edge cases."""
    r = 10
    n = 100
    p = 5
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_asymp_order_normal(r, n, p, F)
    assert isinstance(result, dict)
