"""Tests for grtrc.geron_tree_classification_leaf."""
import numpy as np
import pytest
from morie.fn.grtrc import geron_tree_classification_leaf


def test_grtrc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_classification_leaf(y, leaf_mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtrc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_classification_leaf(y, leaf_mask)
    assert isinstance(result, dict)
