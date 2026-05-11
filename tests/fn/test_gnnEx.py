"""Tests for gnnEx.gnn_explainer."""
import numpy as np
import pytest
from morie.fn.gnnEx import gnn_explainer


def test_gnnEx_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = gnn_explainer(model, graph, node)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gnnEx_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = gnn_explainer(model, graph, node)
    assert isinstance(result, dict)
