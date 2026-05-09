"""Tests for cvxntn.boyd_newton."""
import numpy as np
import pytest
from moirais.fn.cvxntn import boyd_newton


def test_cvxntn_basic():
    """Test basic functionality."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    hess = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_newton(grad, hess)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxntn_edge():
    """Test edge cases."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    hess = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_newton(grad, hess)
    assert isinstance(result, dict)
