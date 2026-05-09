"""Tests for alumap.alammar_umap_projection."""
import numpy as np
import pytest
from moirais.fn.alumap import alammar_umap_projection


def test_alumap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_umap_projection(X, n_neighbors, min_dist, d_out)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alumap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    min_dist = np.random.default_rng(42).normal(0, 1, 100)
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_umap_projection(X, n_neighbors, min_dist, d_out)
    assert isinstance(result, dict)
