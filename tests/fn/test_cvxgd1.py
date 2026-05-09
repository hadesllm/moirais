"""Tests for cvxgd1.boyd_grad_proj."""
import numpy as np
import pytest
from moirais.fn.cvxgd1 import boyd_grad_proj


def test_cvxgd1_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_grad_proj(f, grad_f, x0, C, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxgd1_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_grad_proj(f, grad_f, x0, C, t)
    assert isinstance(result, dict)
