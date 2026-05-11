"""Tests for rgsprep.rangayyan_sparse_rep."""
import numpy as np
import pytest
from morie.fn.rgsprep import rangayyan_sparse_rep


def test_rgsprep_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    lambda_or_sparsity = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = rangayyan_sparse_rep(x, D, lambda_or_sparsity, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsprep_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    lambda_or_sparsity = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = rangayyan_sparse_rep(x, D, lambda_or_sparsity, method)
    assert isinstance(result, dict)
