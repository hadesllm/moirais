"""Tests for km015.kamath_ch2_multihead_head_i."""
import numpy as np
import pytest
from moirais.fn.km015 import kamath_ch2_multihead_head_i


def test_km015_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    W_Qi = np.random.default_rng(42).normal(0, 1, 100)
    W_Ki = np.random.default_rng(42).normal(0, 1, 100)
    W_Vi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_multihead_head_i(Q, K, V, W_Qi, W_Ki, W_Vi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km015_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    W_Qi = np.random.default_rng(42).normal(0, 1, 100)
    W_Ki = np.random.default_rng(42).normal(0, 1, 100)
    W_Vi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_multihead_head_i(Q, K, V, W_Qi, W_Ki, W_Vi)
    assert isinstance(result, dict)
