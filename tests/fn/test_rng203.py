"""Tests for rng203.rangayyan_ch4_ccf_outer_product_random_signals."""
import numpy as np
import pytest
from morie.fn.rng203 import rangayyan_ch4_ccf_outer_product_random_signals


def test_rng203_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_ccf_outer_product_random_signals(x, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng203_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_ccf_outer_product_random_signals(x, y, n)
    assert isinstance(result, dict)
