"""Tests for ginemd.graph_isomorphism_net."""
import numpy as np
import pytest
from morie.fn.ginemd import graph_isomorphism_net


def test_ginemd_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = graph_isomorphism_net(G, X, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ginemd_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = graph_isomorphism_net(G, X, eps)
    assert isinstance(result, dict)
