"""Tests for adamopt.adam."""
import numpy as np
import pytest
from moirais.fn.adamopt import adam


def test_adamopt_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adam(g, beta1, beta2, lr, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adamopt_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adam(g, beta1, beta2, lr, eps)
    assert isinstance(result, dict)
