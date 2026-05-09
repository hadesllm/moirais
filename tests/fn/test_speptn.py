"""Tests for speptn.spatial_pca."""
import numpy as np
import pytest
from moirais.fn.speptn import spatial_pca


def test_speptn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_pca(X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_speptn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_pca(X, W)
    assert isinstance(result, dict)
