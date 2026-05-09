"""Tests for alswa.alammar_sliding_window_attention."""
import numpy as np
import pytest
from moirais.fn.alswa import alammar_sliding_window_attention


def test_alswa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    result = alammar_sliding_window_attention(Q, K, V, window_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alswa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    result = alammar_sliding_window_attention(Q, K, V, window_size)
    assert isinstance(result, dict)
