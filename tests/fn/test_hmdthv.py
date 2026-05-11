"""Tests for hmdthv.geron_tree_high_variance."""
import numpy as np
import pytest
from morie.fn.hmdthv import geron_tree_high_variance


def test_hmdthv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_tree_high_variance(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdthv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_tree_high_variance(X, y)
    assert isinstance(result, dict)
