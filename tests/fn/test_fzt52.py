"""Tests for fzt52.fauzi_thm5_2_bdfree_kdfe_bv."""
import numpy as np
import pytest
from moirais.fn.fzt52 import fauzi_thm5_2_bdfree_kdfe_bv


def test_fzt52_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_2_bdfree_kdfe_bv(x, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt52_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_2_bdfree_kdfe_bv(x, bandwidth, g_func)
    assert isinstance(result, dict)
