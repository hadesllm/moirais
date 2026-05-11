"""Tests for hmkmpp.geron_kmeans_plus_plus."""
import numpy as np
import pytest
from morie.fn.hmkmpp import geron_kmeans_plus_plus


def test_hmkmpp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kmeans_plus_plus(X, n_clusters, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkmpp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kmeans_plus_plus(X, n_clusters, seed)
    assert isinstance(result, dict)
