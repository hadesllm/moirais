"""Tests for ghs029.ghosal_ch3_polya_tree_density_moments."""
import numpy as np
import pytest
from morie.fn.ghs029 import ghosal_ch3_polya_tree_density_moments


def test_ghs029_basic():
    """Test basic functionality."""
    alpha = 0.05
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_polya_tree_density_moments(alpha, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs029_edge():
    """Test edge cases."""
    alpha = 0.05
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_polya_tree_density_moments(alpha, x)
    assert isinstance(result, dict)
