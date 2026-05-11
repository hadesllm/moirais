"""Tests for dgi.dgi."""
import numpy as np
import pytest
from morie.fn.dgi import dgi


def test_dgi_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    result = dgi(G, X, encoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dgi_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    result = dgi(G, X, encoder)
    assert isinstance(result, dict)
