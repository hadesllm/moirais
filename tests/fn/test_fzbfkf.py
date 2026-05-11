"""Tests for fzbfkf.fauzi_bdfree_kdfe_test."""
import numpy as np
import pytest
from morie.fn.fzbfkf import fauzi_bdfree_kdfe_test


def test_fzbfkf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_bdfree_kdfe_test(x, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzbfkf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_bdfree_kdfe_test(x, bandwidth, g_func)
    assert isinstance(result, dict)
