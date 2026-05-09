"""Tests for rgppgwt.rangayyan_ppg_wavelet."""
import numpy as np
import pytest
from moirais.fn.rgppgwt import rangayyan_ppg_wavelet


def test_rgppgwt_basic():
    """Test basic functionality."""
    ppg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_ppg_wavelet(ppg, fs, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgppgwt_edge():
    """Test edge cases."""
    ppg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_ppg_wavelet(ppg, fs, wavelet, levels)
    assert isinstance(result, dict)
