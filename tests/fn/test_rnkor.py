"""Tests for rnkor.rank_order_statistics."""
import numpy as np
import pytest
from moirais.fn.rnkor import rank_order_statistics


def test_rnkor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rank_order_statistics(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_rnkor_edge():
    """Test edge cases."""
    result = rank_order_statistics(np.array([1.0]))
    assert result['n'] == 1
