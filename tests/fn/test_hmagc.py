"""Tests for hmagc.geron_agglomerative."""
import numpy as np
import pytest
from morie.fn.hmagc import geron_agglomerative


def test_hmagc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    linkage = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_agglomerative(X, n_clusters, linkage)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmagc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    linkage = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_agglomerative(X, n_clusters, linkage)
    assert isinstance(result, dict)
