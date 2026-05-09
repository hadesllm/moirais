"""Tests for rgksv.rangayyan_ksvd."""
import numpy as np
import pytest
from moirais.fn.rgksv import rangayyan_ksvd


def test_rgksv_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    n_atoms = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ksvd(Y, n_atoms, sparsity, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgksv_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    n_atoms = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ksvd(Y, n_atoms, sparsity, max_iter)
    assert isinstance(result, dict)
