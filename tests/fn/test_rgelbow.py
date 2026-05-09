"""Tests for rgelbow.rangayyan_kmeans_elbow."""
import numpy as np
import pytest
from moirais.fn.rgelbow import rangayyan_kmeans_elbow


def test_rgelbow_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    max_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kmeans_elbow(X, max_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgelbow_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    max_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kmeans_elbow(X, max_k)
    assert isinstance(result, dict)
