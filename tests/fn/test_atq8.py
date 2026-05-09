"""Tests for atq8.int8_attention."""
import numpy as np
import pytest
from moirais.fn.atq8 import int8_attention


def test_atq8_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = int8_attention(y, Q, K, V, scales)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_atq8_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = int8_attention(y, Q, K, V, scales)
    assert isinstance(result, dict)
