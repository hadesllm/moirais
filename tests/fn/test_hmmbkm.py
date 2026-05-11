"""Tests for hmmbkm.geron_minibatch_kmeans."""
import numpy as np
import pytest
from morie.fn.hmmbkm import geron_minibatch_kmeans


def test_hmmbkm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    seed = 42
    result = geron_minibatch_kmeans(X, n_clusters, batch_size, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmbkm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    seed = 42
    result = geron_minibatch_kmeans(X, n_clusters, batch_size, seed)
    assert isinstance(result, dict)
