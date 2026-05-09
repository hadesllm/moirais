"""Tests for gdupd.gradient_descent_update."""
import numpy as np
import pytest
from moirais.fn.gdupd import gradient_descent_update


def test_gdupd_basic():
    """Test basic functionality."""
    beta = 0.8
    grad = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gradient_descent_update(beta, grad, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gdupd_edge():
    """Test edge cases."""
    beta = 0.8
    grad = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gradient_descent_update(beta, grad, alpha)
    assert isinstance(result, dict)
