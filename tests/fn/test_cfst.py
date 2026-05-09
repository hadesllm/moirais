"""Tests for cfst.causal_forest."""
import numpy as np
import pytest
from moirais.fn.cfst import causal_forest


def test_cfst_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    min_node_size = 100
    result = causal_forest(Y, T, X, n_trees, min_node_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cfst_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    min_node_size = 100
    result = causal_forest(Y, T, X, n_trees, min_node_size)
    assert isinstance(result, dict)
