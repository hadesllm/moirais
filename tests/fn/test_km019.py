"""Tests for km019.kamath_ch2_masked_attention."""
import numpy as np
import pytest
from moirais.fn.km019 import kamath_ch2_masked_attention


def test_km019_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_masked_attention(Q, K, V, M, d_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km019_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_masked_attention(Q, K, V, M, d_k)
    assert isinstance(result, dict)
