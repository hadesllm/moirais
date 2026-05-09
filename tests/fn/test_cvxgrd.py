"""Tests for cvxgrd.boyd_gradient_descent."""
import numpy as np
import pytest
from moirais.fn.cvxgrd import boyd_gradient_descent


def test_cvxgrd_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_gradient_descent(f, grad_f, x0, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxgrd_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_gradient_descent(f, grad_f, x0, t)
    assert isinstance(result, dict)
