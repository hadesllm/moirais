"""Test KV-cache."""
import numpy as np
from morie.fn.kvcch import kvcch


def test_kvcch_new_cache():
    """Test creating new cache."""
    k = np.random.randn(5, 64)
    v = np.random.randn(5, 64)
    result = kvcch(k, v)
    assert result["cache_k"].shape == k.shape
    assert result["cache_v"].shape == v.shape


def test_kvcch_append():
    """Test appending to cache."""
    k_old = np.random.randn(5, 64)
    v_old = np.random.randn(5, 64)
    k_new = np.random.randn(1, 64)
    v_new = np.random.randn(1, 64)
    result = kvcch(k_new, v_new, cache_k=k_old, cache_v=v_old)
    assert result["cache_k"].shape == (6, 64)
    assert result["cache_v"].shape == (6, 64)
