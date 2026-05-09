"""Tests for hmiso.geron_isomap."""
import numpy as np
import pytest
from moirais.fn.hmiso import geron_isomap


def test_hmiso_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_isomap(X, n_components, n_neighbors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmiso_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    n_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_isomap(X, n_components, n_neighbors)
    assert isinstance(result, dict)
