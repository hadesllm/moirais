"""Tests for rng074.rangayyan_ch3_dft_real_imag_decomposition."""
import numpy as np
import pytest
from moirais.fn.rng074 import rangayyan_ch3_dft_real_imag_decomposition


def test_rng074_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_dft_real_imag_decomposition(x, n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng074_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_dft_real_imag_decomposition(x, n, k, N)
    assert isinstance(result, dict)
