"""Tests for cvxbgm.boyd_basis_pursuit."""
import numpy as np
import pytest
from morie.fn.cvxbgm import boyd_basis_pursuit


def test_cvxbgm_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_basis_pursuit(A, b, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxbgm_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_basis_pursuit(A, b, eps)
    assert isinstance(result, dict)
