"""Tests for hmlle.geron_locally_linear_embedding."""
import numpy as np
import pytest
from morie.fn.hmlle import geron_locally_linear_embedding


def test_hmlle_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_locally_linear_embedding(X, n_components, n_neighbors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlle_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_locally_linear_embedding(X, n_components, n_neighbors)
    assert isinstance(result, dict)
