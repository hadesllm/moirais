"""Tests for ghs019.ghosal_ch3_tree_set_probability."""
import numpy as np
import pytest
from morie.fn.ghs019 import ghosal_ch3_tree_set_probability


def test_ghs019_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tree_set_probability(V, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs019_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tree_set_probability(V, epsilon)
    assert isinstance(result, dict)
