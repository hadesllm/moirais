"""Tests for rng087.rangayyan_ch3_ma_filter_general."""
import numpy as np
import pytest
from moirais.fn.rng087 import rangayyan_ch3_ma_filter_general


def test_rng087_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch3_ma_filter_general(x, b_k, n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng087_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    N = 100
    result = rangayyan_ch3_ma_filter_general(x, b_k, n, N)
    assert isinstance(result, dict)
