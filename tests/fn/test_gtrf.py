"""Tests for gtrf.graph_transformer."""
import numpy as np
import pytest
from moirais.fn.gtrf import graph_transformer


def test_gtrf_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = graph_transformer(G, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gtrf_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = graph_transformer(G, X)
    assert isinstance(result, dict)
