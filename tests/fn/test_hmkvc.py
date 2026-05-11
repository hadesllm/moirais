"""Tests for hmkvc.geron_kv_cache_compress."""
import numpy as np
import pytest
from morie.fn.hmkvc import geron_kv_cache_compress


def test_hmkvc_basic():
    """Test basic functionality."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compress(K, V, n_bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkvc_edge():
    """Test edge cases."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compress(K, V, n_bits)
    assert isinstance(result, dict)
