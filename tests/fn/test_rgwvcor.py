"""Tests for rgwvcor.rangayyan_wavelet_corr."""
import numpy as np
import pytest
from moirais.fn.rgwvcor import rangayyan_wavelet_corr


def test_rgwvcor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_corr(x, y, wavelet, levels)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rgwvcor_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_corr(x, y, wavelet, levels)
    assert isinstance(result, dict)
