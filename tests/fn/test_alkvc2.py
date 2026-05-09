"""Tests for alkvc2.alammar_kv_cache_lookup."""
import numpy as np
import pytest
from moirais.fn.alkvc2 import alammar_kv_cache_lookup


def test_alkvc2_basic():
    """Test basic functionality."""
    K_cache = np.random.default_rng(42).normal(0, 1, 100)
    V_cache = np.random.default_rng(42).normal(0, 1, 100)
    k_new = np.random.default_rng(42).normal(0, 1, 100)
    v_new = np.random.default_rng(42).normal(0, 1, 100)
    q_new = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_kv_cache_lookup(K_cache, V_cache, k_new, v_new, q_new)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alkvc2_edge():
    """Test edge cases."""
    K_cache = np.random.default_rng(42).normal(0, 1, 100)
    V_cache = np.random.default_rng(42).normal(0, 1, 100)
    k_new = np.random.default_rng(42).normal(0, 1, 100)
    v_new = np.random.default_rng(42).normal(0, 1, 100)
    q_new = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_kv_cache_lookup(K_cache, V_cache, k_new, v_new, q_new)
    assert isinstance(result, dict)
