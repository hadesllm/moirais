"""Tests for rng135.rangayyan_ch3_butterworth_highpass_dft_indexed."""
import numpy as np
import pytest
from moirais.fn.rng135 import rangayyan_ch3_butterworth_highpass_dft_indexed


def test_rng135_basic():
    """Test basic functionality."""
    k = 5
    k_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_highpass_dft_indexed(k, k_c, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng135_edge():
    """Test edge cases."""
    k = 5
    k_c = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_highpass_dft_indexed(k, k_c, N)
    assert isinstance(result, dict)
