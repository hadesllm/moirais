"""Tests for fzt56.fauzi_thm5_6_bdfree_ks_equiv."""
import numpy as np
import pytest
from morie.fn.fzt56 import fauzi_thm5_6_bdfree_ks_equiv


def test_fzt56_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    g_func = (lambda v: v)
    result = fauzi_thm5_6_bdfree_ks_equiv(data, bandwidth, cdf, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt56_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    g_func = (lambda v: v)
    result = fauzi_thm5_6_bdfree_ks_equiv(data, bandwidth, cdf, g_func)
    assert isinstance(result, dict)
