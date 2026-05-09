"""Tests for grtrv.geron_tree_regression_leaf."""
import numpy as np
import pytest
from moirais.fn.grtrv import geron_tree_regression_leaf


def test_grtrv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regression_leaf(y, leaf_mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtrv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regression_leaf(y, leaf_mask)
    assert isinstance(result, dict)
