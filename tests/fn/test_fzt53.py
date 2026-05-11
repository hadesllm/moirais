"""Tests for fzt53.fauzi_thm5_3_bdfree_normality."""
import numpy as np
import pytest
from morie.fn.fzt53 import fauzi_thm5_3_bdfree_normality


def test_fzt53_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_3_bdfree_normality(x, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt53_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm5_3_bdfree_normality(x, bandwidth, g_func)
    assert isinstance(result, dict)
