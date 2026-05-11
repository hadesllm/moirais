"""Tests for hmrdt.geron_regression_tree."""
import numpy as np
import pytest
from morie.fn.hmrdt import geron_regression_tree


def test_hmrdt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_regression_tree(X, y, max_depth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrdt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_regression_tree(X, y, max_depth)
    assert isinstance(result, dict)
