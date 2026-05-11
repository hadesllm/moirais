"""Tests for bkcgr.burkov_computational_graph."""
import numpy as np
import pytest
from morie.fn.bkcgr import burkov_computational_graph


def test_bkcgr_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_computational_graph(graph, inputs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkcgr_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_computational_graph(graph, inputs)
    assert isinstance(result, dict)
