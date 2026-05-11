"""Tests for hetgnn.heterogeneous_gnn."""
import numpy as np
import pytest
from morie.fn.hetgnn import heterogeneous_gnn


def test_hetgnn_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    metapaths = np.random.default_rng(42).normal(0, 1, 100)
    result = heterogeneous_gnn(G, X, metapaths)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hetgnn_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    metapaths = np.random.default_rng(42).normal(0, 1, 100)
    result = heterogeneous_gnn(G, X, metapaths)
    assert isinstance(result, dict)
