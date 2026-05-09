"""Tests for rgomp.rangayyan_omp."""
import numpy as np
import pytest
from moirais.fn.rgomp import rangayyan_omp


def test_rgomp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_omp(x, D, sparsity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgomp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_omp(x, D, sparsity)
    assert isinstance(result, dict)
