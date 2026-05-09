"""Tests for rmspO.rmsprop_optimizer."""
import numpy as np
import pytest
from moirais.fn.rmspO import rmsprop_optimizer


def test_rmspO_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsprop_optimizer(theta, grad, lr, rho, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rmspO_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsprop_optimizer(theta, grad, lr, rho, eps)
    assert isinstance(result, dict)
