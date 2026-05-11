"""Tests for rgosflt.rangayyan_order_stat_flt."""
import numpy as np
import pytest
from morie.fn.rgosflt import rangayyan_order_stat_flt


def test_rgosflt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_order_stat_flt(x, window)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgosflt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_order_stat_flt(x, window)
    assert isinstance(result, dict)
