"""Tests for gradmo.geron_adam_update."""
import numpy as np
import pytest
from moirais.fn.gradmo import geron_adam_update


def test_gradmo_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    s = 90
    t = np.linspace(0, 10, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adam_update(theta, grad, m, s, t, eta, b1, b2, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gradmo_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    s = 90
    t = np.linspace(0, 10, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adam_update(theta, grad, m, s, t, eta, b1, b2, eps)
    assert isinstance(result, dict)
