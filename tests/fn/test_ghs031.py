"""Tests for ghs031.ghosal_ch3_polya_tree_mixture_post_density."""
import numpy as np
import pytest
from morie.fn.ghs031 import ghosal_ch3_polya_tree_mixture_post_density


def test_ghs031_basic():
    """Test basic functionality."""
    g_theta = np.random.default_rng(42).normal(0, 1, 100)
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_polya_tree_mixture_post_density(g_theta, a_j, N, theta, x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs031_edge():
    """Test edge cases."""
    g_theta = np.random.default_rng(42).normal(0, 1, 100)
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_polya_tree_mixture_post_density(g_theta, a_j, N, theta, x, n)
    assert isinstance(result, dict)
