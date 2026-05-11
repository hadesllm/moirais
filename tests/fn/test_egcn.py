"""Tests for egcn.e_gcn."""
import numpy as np
import pytest
from morie.fn.egcn import e_gcn


def test_egcn_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = e_gcn(G, X, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egcn_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = e_gcn(G, X, coords)
    assert isinstance(result, dict)
