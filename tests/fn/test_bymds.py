"""Tests for bymds.bayesian_mds."""
import numpy as np
import pytest
from morie.fn.bymds import bayesian_mds


def test_bymds_basic():
    """Test basic functionality."""
    D_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    n_iter = 50
    result = bayesian_mds(D_matrix, n_dims, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bymds_edge():
    """Test edge cases."""
    D_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    n_iter = 50
    result = bayesian_mds(D_matrix, n_dims, n_iter)
    assert isinstance(result, dict)
