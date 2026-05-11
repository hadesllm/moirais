"""Tests for grkvc.geron_kv_cache_compression."""
import numpy as np
import pytest
from morie.fn.grkvc import geron_kv_cache_compression


def test_grkvc_basic():
    """Test basic functionality."""
    seq_len = 100
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_head = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkvc_edge():
    """Test edge cases."""
    seq_len = 100
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_head = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits)
    assert isinstance(result, dict)
