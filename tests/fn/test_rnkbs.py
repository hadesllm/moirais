"""Tests for rnkbs.rank_based_test."""
import numpy as np
import pytest
from morie.fn.rnkbs import rank_based_test


def test_rnkbs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rank_based_test(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_rnkbs_edge():
    """Test edge cases."""
    result = rank_based_test(np.array([1.0]))
    assert result['n'] == 1
