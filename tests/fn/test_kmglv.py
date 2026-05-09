"""Tests for kmglv.kamath_glove_cost."""
import numpy as np
import pytest
from moirais.fn.kmglv import kamath_glove_cost


def test_kmglv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    W_tilde = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    b_tilde = np.random.default_rng(42).normal(0, 1, 100)
    x_max = 100
    alpha = 0.05
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmglv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    W_tilde = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    b_tilde = np.random.default_rng(42).normal(0, 1, 100)
    x_max = 100
    alpha = 0.05
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max, alpha)
    assert isinstance(result, dict)
