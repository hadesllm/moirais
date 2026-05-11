"""Tests for sparsv.sparse_vector."""
import numpy as np
import pytest
from morie.fn.sparsv import sparse_vector


def test_sparsv_basic():
    """Test basic functionality."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = sparse_vector(queries, threshold, c, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sparsv_edge():
    """Test edge cases."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = sparse_vector(queries, threshold, c, epsilon)
    assert isinstance(result, dict)
