"""Tests for slbpdg.slice_break_dp."""
import numpy as np
import pytest
from morie.fn.slbpdg import slice_break_dp


def test_slbpdg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = slice_break_dp(y, alpha, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_slbpdg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = slice_break_dp(y, alpha, n_iter)
    assert isinstance(result, dict)
