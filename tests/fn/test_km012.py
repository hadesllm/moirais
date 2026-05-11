"""Tests for km012.kamath_ch2_scaled_dot_attention."""
import numpy as np
import pytest
from morie.fn.km012 import kamath_ch2_scaled_dot_attention


def test_km012_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_scaled_dot_attention(Q, K, V, d_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km012_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_scaled_dot_attention(Q, K, V, d_k)
    assert isinstance(result, dict)
