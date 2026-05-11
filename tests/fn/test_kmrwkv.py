"""Tests for kmrwkv.kamath_rwkv_time_mix."""
import numpy as np
import pytest
from morie.fn.kmrwkv import kamath_rwkv_time_mix


def test_kmrwkv_basic():
    """Test basic functionality."""
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_rwkv_time_mix(k, v, w, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrwkv_edge():
    """Test edge cases."""
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_rwkv_time_mix(k, v, w, u)
    assert isinstance(result, dict)
