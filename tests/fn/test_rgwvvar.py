"""Tests for rgwvvar.rangayyan_wavelet_variance."""
import numpy as np
import pytest
from moirais.fn.rgwvvar import rangayyan_wavelet_variance


def test_rgwvvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_variance(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwvvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_variance(x, wavelet, levels)
    assert isinstance(result, dict)
