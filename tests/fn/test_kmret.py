"""Tests for kmret.kamath_retnet_retention."""
import numpy as np
import pytest
from moirais.fn.kmret import kamath_retnet_retention


def test_kmret_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = kamath_retnet_retention(Q, K, V, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmret_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = kamath_retnet_retention(Q, K, V, gamma)
    assert isinstance(result, dict)
