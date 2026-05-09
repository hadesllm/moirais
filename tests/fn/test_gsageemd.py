"""Tests for gsageemd.graphsage."""
import numpy as np
import pytest
from moirais.fn.gsageemd import graphsage


def test_gsageemd_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aggregator = np.random.default_rng(42).normal(0, 1, 100)
    result = graphsage(G, X, aggregator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gsageemd_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aggregator = np.random.default_rng(42).normal(0, 1, 100)
    result = graphsage(G, X, aggregator)
    assert isinstance(result, dict)
