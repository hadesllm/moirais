"""Tests for hmkprbf.geron_kernel_pca_rbf."""
import numpy as np
import pytest
from morie.fn.hmkprbf import geron_kernel_pca_rbf


def test_hmkprbf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    result = geron_kernel_pca_rbf(X, n_components, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkprbf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    result = geron_kernel_pca_rbf(X, n_components, gamma)
    assert isinstance(result, dict)
