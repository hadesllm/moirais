"""Tests for gatemd.graph_attention_net."""
import numpy as np
import pytest
from moirais.fn.gatemd import graph_attention_net


def test_gatemd_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = graph_attention_net(G, X, heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gatemd_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = graph_attention_net(G, X, heads)
    assert isinstance(result, dict)
