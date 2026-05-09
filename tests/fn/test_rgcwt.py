"""Tests for rgcwt.rangayyan_cwt."""
import numpy as np
import pytest
from moirais.fn.rgcwt import rangayyan_cwt


def test_rgcwt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    wavelet = 'morl'
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cwt(x, fs, wavelet, scales)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcwt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    wavelet = 'morl'
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cwt(x, fs, wavelet, scales)
    assert isinstance(result, dict)
