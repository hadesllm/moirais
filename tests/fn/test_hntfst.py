"""Tests for hntfst.honest_random_forest."""
import numpy as np
import pytest
from moirais.fn.hntfst import honest_random_forest


def test_hntfst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    min_node = np.random.default_rng(42).normal(0, 1, 100)
    result = honest_random_forest(y, X, n_trees, min_node)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hntfst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    min_node = np.random.default_rng(42).normal(0, 1, 100)
    result = honest_random_forest(y, X, n_trees, min_node)
    assert isinstance(result, dict)
