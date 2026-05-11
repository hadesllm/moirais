"""Tests for almqa.alammar_multi_query_attention."""
import numpy as np
import pytest
from morie.fn.almqa import alammar_multi_query_attention


def test_almqa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K_shared = np.random.default_rng(42).normal(0, 1, 100)
    V_shared = np.random.default_rng(42).normal(0, 1, 100)
    n_query_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_multi_query_attention(Q, K_shared, V_shared, n_query_heads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almqa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K_shared = np.random.default_rng(42).normal(0, 1, 100)
    V_shared = np.random.default_rng(42).normal(0, 1, 100)
    n_query_heads = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_multi_query_attention(Q, K_shared, V_shared, n_query_heads)
    assert isinstance(result, dict)
