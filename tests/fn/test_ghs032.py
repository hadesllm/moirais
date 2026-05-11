"""Tests for ghs032.ghosal_ch3_polya_tree_density_bounds."""
import numpy as np
import pytest
from morie.fn.ghs032 import ghosal_ch3_polya_tree_density_bounds


def test_ghs032_basic():
    """Test basic functionality."""
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    m = 10
    result = ghosal_ch3_polya_tree_density_bounds(a_j, n, N, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs032_edge():
    """Test edge cases."""
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    m = 10
    result = ghosal_ch3_polya_tree_density_bounds(a_j, n, N, m)
    assert isinstance(result, dict)
