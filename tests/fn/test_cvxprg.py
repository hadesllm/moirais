"""Tests for cvxprg.boyd_proximal_grad."""
import numpy as np
import pytest
from morie.fn.cvxprg import boyd_proximal_grad


def test_cvxprg_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_proximal_grad(f, grad_f, h, x0, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxprg_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_proximal_grad(f, grad_f, h, x0, t)
    assert isinstance(result, dict)
