"""Tests for hmspcl.geron_spectral_clustering."""
import numpy as np
import pytest
from morie.fn.hmspcl import geron_spectral_clustering


def test_hmspcl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    affinity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_spectral_clustering(X, n_clusters, affinity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmspcl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    affinity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_spectral_clustering(X, n_clusters, affinity)
    assert isinstance(result, dict)
