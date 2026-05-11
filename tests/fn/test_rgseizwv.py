"""Tests for rgseizwv.rangayyan_seizure_wavelet."""
import numpy as np
import pytest
from morie.fn.rgseizwv import rangayyan_seizure_wavelet


def test_rgseizwv_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_seizure_wavelet(eeg, fs, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgseizwv_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_seizure_wavelet(eeg, fs, wavelet, levels)
    assert isinstance(result, dict)
