"""Tests for scumap.umap_singlecell."""
import numpy as np
import pytest
from morie.fn.scumap import umap_singlecell


def test_scumap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = umap_singlecell(X, n_neighbors, min_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scumap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = umap_singlecell(X, n_neighbors, min_dist)
    assert isinstance(result, dict)
