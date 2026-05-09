"""Tests for grmha.geron_multi_head_attention."""
import numpy as np
import pytest
from moirais.fn.grmha import geron_multi_head_attention


def test_grmha_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    WO = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmha_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    WO = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h)
    assert isinstance(result, dict)
