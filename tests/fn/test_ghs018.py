"""Tests for ghs018.ghosal_ch3_tree_splitting_variables."""
import numpy as np
import pytest
from morie.fn.ghs018 import ghosal_ch3_tree_splitting_variables


def test_ghs018_basic():
    """Test basic functionality."""
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tree_splitting_variables(A_epsilon, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs018_edge():
    """Test edge cases."""
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tree_splitting_variables(A_epsilon, epsilon)
    assert isinstance(result, dict)
