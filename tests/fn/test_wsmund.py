"""Tests for wsmund.wasserman_undirected_graph."""
import numpy as np
import pytest
from morie.fn.wsmund import wasserman_undirected_graph


def test_wsmund_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_undirected_graph(graph, psi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmund_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_undirected_graph(graph, psi)
    assert isinstance(result, dict)
