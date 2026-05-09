"""Tests for rgwavstr.rangayyan_wavelet_struct."""
import numpy as np
import pytest
from moirais.fn.rgwavstr import rangayyan_wavelet_struct


def test_rgwavstr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    scales = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    result = rangayyan_wavelet_struct(x, fs, scales, wavelet)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwavstr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    scales = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    result = rangayyan_wavelet_struct(x, fs, scales, wavelet)
    assert isinstance(result, dict)
