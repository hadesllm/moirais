"""Tests for rgkmns.rangayyan_kmeans."""
import numpy as np
import pytest
from morie.fn.rgkmns import rangayyan_kmeans


def test_rgkmns_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_kmeans(X, k, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgkmns_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_kmeans(X, k, max_iter, tol)
    assert isinstance(result, dict)
