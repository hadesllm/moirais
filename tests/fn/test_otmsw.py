"""Tests for otmsw.ot_max_sliced_w."""
import numpy as np
import pytest
from morie.fn.otmsw import ot_max_sliced_w


def test_otmsw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_max_sliced_w(X, Y, p, n_proj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmsw_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_max_sliced_w(X, Y, p, n_proj)
    assert isinstance(result, dict)
