"""Tests for hmkmn.geron_kmeans."""
import numpy as np
import pytest
from moirais.fn.hmkmn import geron_kmeans


def test_hmkmn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kmeans(X, n_clusters, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkmn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kmeans(X, n_clusters, seed)
    assert isinstance(result, dict)
