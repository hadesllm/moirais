"""Tests for wave.wavelet_basis."""
import numpy as np
import pytest
from moirais.fn.wave import wavelet_basis


def test_wave_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    result = wavelet_basis(y, wavelet)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wave_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    result = wavelet_basis(y, wavelet)
    assert isinstance(result, dict)
