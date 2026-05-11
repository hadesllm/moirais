"""Tests for grrmsp.geron_rmsprop_update."""
import numpy as np
import pytest
from morie.fn.grrmsp import geron_rmsprop_update


def test_grrmsp_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rmsprop_update(theta, grad, s, eta, rho, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrmsp_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rmsprop_update(theta, grad, s, eta, rho, eps)
    assert isinstance(result, dict)
