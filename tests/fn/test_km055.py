"""Tests for km055.kamath_ch4_parallel_adapter."""
import numpy as np
import pytest
from morie.fn.km055 import kamath_ch4_parallel_adapter


def test_km055_basic():
    """Test basic functionality."""
    H_o = np.random.default_rng(42).normal(0, 1, 100)
    H_i = np.random.default_rng(42).normal(0, 1, 100)
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_parallel_adapter(H_o, H_i, W_down, W_up)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km055_edge():
    """Test edge cases."""
    H_o = np.random.default_rng(42).normal(0, 1, 100)
    H_i = np.random.default_rng(42).normal(0, 1, 100)
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_parallel_adapter(H_o, H_i, W_down, W_up)
    assert isinstance(result, dict)
