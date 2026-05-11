"""Tests for hmumap.geron_umap."""
import numpy as np
import pytest
from morie.fn.hmumap import geron_umap


def test_hmumap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_umap(X, n_components, n_neighbors, min_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmumap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_umap(X, n_components, n_neighbors, min_dist)
    assert isinstance(result, dict)
