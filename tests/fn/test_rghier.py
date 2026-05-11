"""Tests for rghier.rangayyan_hierarchical_clust."""
import numpy as np
import pytest
from morie.fn.rghier import rangayyan_hierarchical_clust


def test_rghier_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    linkage = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hierarchical_clust(X, linkage, n_clusters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghier_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    linkage = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hierarchical_clust(X, linkage, n_clusters)
    assert isinstance(result, dict)
