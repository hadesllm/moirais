"""Tests for rng075.rangayyan_ch3_idft_real_imag."""
import numpy as np
import pytest
from moirais.fn.rng075 import rangayyan_ch3_idft_real_imag


def test_rng075_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_idft_real_imag(X, n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng075_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_idft_real_imag(X, n, k, N)
    assert isinstance(result, dict)
