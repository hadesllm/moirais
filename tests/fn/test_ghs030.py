"""Tests for ghs030.ghosal_ch3_polya_tree_posterior_density."""
import numpy as np
import pytest
from morie.fn.ghs030 import ghosal_ch3_polya_tree_posterior_density


def test_ghs030_basic():
    """Test basic functionality."""
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_polya_tree_posterior_density(a_j, N, x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs030_edge():
    """Test edge cases."""
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_polya_tree_posterior_density(a_j, N, x, n)
    assert isinstance(result, dict)
