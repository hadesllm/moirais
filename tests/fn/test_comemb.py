"""Tests for comemb.node2vec."""
import numpy as np
import pytest
from moirais.fn.comemb import node2vec


def test_comemb_basic():
    """Test basic functionality."""
    G = np.eye(10)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = node2vec(G, p, q, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comemb_edge():
    """Test edge cases."""
    G = np.eye(10)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = node2vec(G, p, q, dim)
    assert isinstance(result, dict)
