"""Tests for rgldsp.rangayyan_dictionary_sparse."""
import numpy as np
import pytest
from moirais.fn.rgldsp import rangayyan_dictionary_sparse


def test_rgldsp_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sparsity_T = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_dictionary_sparse(Y, D, sparsity_T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgldsp_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sparsity_T = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_dictionary_sparse(Y, D, sparsity_T)
    assert isinstance(result, dict)
