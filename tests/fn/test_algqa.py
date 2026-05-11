"""Tests for algqa.alammar_grouped_query_attention."""
import numpy as np
import pytest
from morie.fn.algqa import alammar_grouped_query_attention


def test_algqa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_query_heads = np.random.default_rng(42).normal(0, 1, 100)
    n_kv_groups = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_grouped_query_attention(Q, K, V, n_query_heads, n_kv_groups)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_algqa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_query_heads = np.random.default_rng(42).normal(0, 1, 100)
    n_kv_groups = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_grouped_query_attention(Q, K, V, n_query_heads, n_kv_groups)
    assert isinstance(result, dict)
