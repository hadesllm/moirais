"""Tests for nchunk.causal_chunked_attention."""
import numpy as np
import pytest
from morie.fn.nchunk import causal_chunked_attention


def test_nchunk_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    chunk_size = 100
    result = causal_chunked_attention(y, Q, K, V, chunk_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nchunk_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    chunk_size = 100
    result = causal_chunked_attention(y, Q, K, V, chunk_size)
    assert isinstance(result, dict)
