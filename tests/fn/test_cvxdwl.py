"""Tests for cvxdwl.boyd_dual_function."""
import numpy as np
import pytest
from moirais.fn.cvxdwl import boyd_dual_function


def test_cvxdwl_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_dual_function(L, lambda_, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxdwl_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_dual_function(L, lambda_, nu)
    assert isinstance(result, dict)
