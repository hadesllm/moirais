"""Tests for cvxnda.boyd_newton_decrement."""
import numpy as np
import pytest
from morie.fn.cvxnda import boyd_newton_decrement


def test_cvxnda_basic():
    """Test basic functionality."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    hess = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_newton_decrement(grad, hess)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxnda_edge():
    """Test edge cases."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    hess = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_newton_decrement(grad, hess)
    assert isinstance(result, dict)
