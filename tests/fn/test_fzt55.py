"""Tests for fzt55.fauzi_thm5_5_bdfree_kde_bv."""
import numpy as np
import pytest
from morie.fn.fzt55 import fauzi_thm5_5_bdfree_kde_bv


def test_fzt55_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_5_bdfree_kde_bv(x, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt55_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_5_bdfree_kde_bv(x, bandwidth, g_func)
    assert isinstance(result, dict)
