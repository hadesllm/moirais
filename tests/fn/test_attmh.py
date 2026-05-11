"""Tests for attmh.multi_head_attention."""
import numpy as np
import pytest
from morie.fn.attmh import multi_head_attention


def test_attmh_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    Wq = np.random.default_rng(42).normal(0, 1, 100)
    Wk = np.random.default_rng(42).normal(0, 1, 100)
    Wv = np.random.default_rng(42).normal(0, 1, 100)
    Wo = np.random.default_rng(42).normal(0, 1, 100)
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = multi_head_attention(y, Q, K, V, Wq, Wk, Wv, Wo, heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_attmh_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    Wq = np.random.default_rng(42).normal(0, 1, 100)
    Wk = np.random.default_rng(42).normal(0, 1, 100)
    Wv = np.random.default_rng(42).normal(0, 1, 100)
    Wo = np.random.default_rng(42).normal(0, 1, 100)
    heads = np.random.default_rng(42).normal(0, 1, 100)
    result = multi_head_attention(y, Q, K, V, Wq, Wk, Wv, Wo, heads)
    assert isinstance(result, dict)
