"""Tests for grada2.geron_adagrad_update."""
import numpy as np
import pytest
from morie.fn.grada2 import geron_adagrad_update


def test_grada2_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adagrad_update(theta, grad, s, eta, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grada2_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adagrad_update(theta, grad, s, eta, eps)
    assert isinstance(result, dict)
