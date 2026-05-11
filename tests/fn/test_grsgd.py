"""Tests for grsgd.geron_stochastic_gradient_descent."""
import numpy as np
import pytest
from morie.fn.grsgd import geron_stochastic_gradient_descent


def test_grsgd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsgd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)
