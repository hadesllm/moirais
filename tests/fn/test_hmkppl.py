"""Tests for hmkppl.geron_kernel_pca_poly."""
import numpy as np
import pytest
from moirais.fn.hmkppl import geron_kernel_pca_poly


def test_hmkppl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    degree = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    coef0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kernel_pca_poly(X, n_components, degree, gamma, coef0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkppl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    degree = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    coef0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kernel_pca_poly(X, n_components, degree, gamma, coef0)
    assert isinstance(result, dict)
