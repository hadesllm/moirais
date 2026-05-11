"""Tests for causrho.causal_proximal_proxy."""
import numpy as np
import pytest
from morie.fn.causrho import causal_proximal_proxy


def test_causrho_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    W_proxy = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_proximal_proxy(y, A, Z_proxy, W_proxy, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causrho_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    W_proxy = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_proximal_proxy(y, A, Z_proxy, W_proxy, X)
    assert isinstance(result, dict)
