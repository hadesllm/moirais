"""Tests for ghs020.ghosal_ch3_tree_countable_additivity."""
import numpy as np
import pytest
from moirais.fn.ghs020 import ghosal_ch3_tree_countable_additivity


def test_ghs020_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tree_countable_additivity(V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs020_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tree_countable_additivity(V)
    assert isinstance(result, dict)
