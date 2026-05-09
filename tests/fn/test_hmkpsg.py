"""Tests for hmkpsg.geron_kernel_pca_sigmoid."""
import numpy as np
import pytest
from moirais.fn.hmkpsg import geron_kernel_pca_sigmoid


def test_hmkpsg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    coef0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kernel_pca_sigmoid(X, n_components, gamma, coef0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkpsg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    gamma = 1.0
    coef0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kernel_pca_sigmoid(X, n_components, gamma, coef0)
    assert isinstance(result, dict)
