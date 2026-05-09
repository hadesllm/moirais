"""Tests for hmsrp.geron_sparse_rand_projection."""
import numpy as np
import pytest
from moirais.fn.hmsrp import geron_sparse_rand_projection


def test_hmsrp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    density = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_sparse_rand_projection(X, d_out, density, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsrp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    density = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_sparse_rand_projection(X, d_out, density, seed)
    assert isinstance(result, dict)
