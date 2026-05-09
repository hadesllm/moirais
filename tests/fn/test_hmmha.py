"""Tests for hmmha.geron_multihead_attention."""
import numpy as np
import pytest
from moirais.fn.hmmha import geron_multihead_attention


def test_hmmha_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    W_O = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_multihead_attention(Q, K, V, n_heads, W_O)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmha_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_heads = np.random.default_rng(42).normal(0, 1, 100)
    W_O = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_multihead_attention(Q, K, V, n_heads, W_O)
    assert isinstance(result, dict)
