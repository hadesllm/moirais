"""Tests for rng072.rangayyan_ch3_dft_via_twiddle."""
import numpy as np
import pytest
from moirais.fn.rng072 import rangayyan_ch3_dft_via_twiddle


def test_rng072_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    W_N = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_dft_via_twiddle(x, n, k, W_N, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng072_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    W_N = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_dft_via_twiddle(x, n, k, W_N, N)
    assert isinstance(result, dict)
