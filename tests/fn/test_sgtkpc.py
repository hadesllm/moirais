"""Tests for sgtkpc.sgt_kernel_pca."""
import numpy as np
import pytest
from morie.fn.sgtkpc import sgt_kernel_pca


def test_sgtkpc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    k = 5
    result = sgt_kernel_pca(X, kernel, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtkpc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    k = 5
    result = sgt_kernel_pca(X, kernel, k)
    assert isinstance(result, dict)
