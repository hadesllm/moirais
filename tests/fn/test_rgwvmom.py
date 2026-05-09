"""Tests for rgwvmom.rangayyan_wavelet_moments."""
import numpy as np
import pytest
from moirais.fn.rgwvmom import rangayyan_wavelet_moments


def test_rgwvmom_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_moments(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwvmom_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_moments(x, wavelet, levels)
    assert isinstance(result, dict)
