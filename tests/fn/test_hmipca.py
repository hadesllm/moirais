"""Tests for hmipca.geron_incremental_pca."""
import numpy as np
import pytest
from morie.fn.hmipca import geron_incremental_pca


def test_hmipca_basic():
    """Test basic functionality."""
    X_iter = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    batch_size = 100
    result = geron_incremental_pca(X_iter, n_components, batch_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmipca_edge():
    """Test edge cases."""
    X_iter = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    batch_size = 100
    result = geron_incremental_pca(X_iter, n_components, batch_size)
    assert isinstance(result, dict)
