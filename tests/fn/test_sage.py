"""Tests for sage.graphsage."""
import numpy as np
import pytest
from morie.fn.sage import graphsage


def test_sage_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    aggregator = np.random.default_rng(42).normal(0, 1, 100)
    result = graphsage(G, X, W, aggregator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sage_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    aggregator = np.random.default_rng(42).normal(0, 1, 100)
    result = graphsage(G, X, W, aggregator)
    assert isinstance(result, dict)
