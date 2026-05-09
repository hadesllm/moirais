"""Tests for retnet.retnet_retention."""
import numpy as np
import pytest
from moirais.fn.retnet import retnet_retention


def test_retnet_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = retnet_retention(y, Q, K, V, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_retnet_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = retnet_retention(y, Q, K, V, gamma)
    assert isinstance(result, dict)
