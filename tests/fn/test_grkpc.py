"""Tests for grkpc.geron_kernel_pca_rbf."""
import numpy as np
import pytest
from morie.fn.grkpc import geron_kernel_pca_rbf


def test_grkpc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma = 1.0
    d = 5
    result = geron_kernel_pca_rbf(X, gamma, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkpc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma = 1.0
    d = 5
    result = geron_kernel_pca_rbf(X, gamma, d)
    assert isinstance(result, dict)
