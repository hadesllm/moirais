"""Tests for cvxrgl.boyd_regularized_ls."""
import numpy as np
import pytest
from morie.fn.cvxrgl import boyd_regularized_ls


def test_cvxrgl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_regularized_ls(A, b, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxrgl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_regularized_ls(A, b, delta)
    assert isinstance(result, dict)
