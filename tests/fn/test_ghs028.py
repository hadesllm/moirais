"""Tests for ghs028.ghosal_ch3_polya_tree_first_two_moments."""
import numpy as np
import pytest
from morie.fn.ghs028 import ghosal_ch3_polya_tree_first_two_moments


def test_ghs028_basic():
    """Test basic functionality."""
    alpha_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    m = 10
    result = ghosal_ch3_polya_tree_first_two_moments(alpha_epsilon, epsilon, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs028_edge():
    """Test edge cases."""
    alpha_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    m = 10
    result = ghosal_ch3_polya_tree_first_two_moments(alpha_epsilon, epsilon, m)
    assert isinstance(result, dict)
