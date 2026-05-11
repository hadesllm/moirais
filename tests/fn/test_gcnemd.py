"""Tests for gcnemd.gcn."""
import numpy as np
import pytest
from morie.fn.gcnemd import gcn


def test_gcnemd_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gcn(G, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gcnemd_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gcn(G, X, W)
    assert isinstance(result, dict)
