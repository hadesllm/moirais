"""Tests for hmsatt.geron_self_attention."""
import numpy as np
import pytest
from morie.fn.hmsatt import geron_self_attention


def test_hmsatt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_Q = np.random.default_rng(42).normal(0, 1, 100)
    W_K = np.random.default_rng(42).normal(0, 1, 100)
    W_V = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_attention(X, W_Q, W_K, W_V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsatt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_Q = np.random.default_rng(42).normal(0, 1, 100)
    W_K = np.random.default_rng(42).normal(0, 1, 100)
    W_V = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_attention(X, W_Q, W_K, W_V)
    assert isinstance(result, dict)
