"""Tests for grkmo.geron_kmeans_objective."""
import numpy as np
import pytest
from morie.fn.grkmo import geron_kmeans_objective


def test_grkmo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    centroids = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkmo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    centroids = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)
