"""Tests for gradds.gradient_descent."""
import numpy as np
import pytest
from morie.fn.gradds import gradient_descent


def test_gradds_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = gradient_descent(f, grad_f, x0, lr, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gradds_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = gradient_descent(f, grad_f, x0, lr, steps)
    assert isinstance(result, dict)
