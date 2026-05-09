"""Tests for adamO.adam_optimizer."""
import numpy as np
import pytest
from moirais.fn.adamO import adam_optimizer


def test_adamO_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adam_optimizer(theta, grad, lr, beta1, beta2, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adamO_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adam_optimizer(theta, grad, lr, beta1, beta2, eps)
    assert isinstance(result, dict)
