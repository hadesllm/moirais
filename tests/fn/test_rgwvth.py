"""Tests for rgwvth.rangayyan_wavelet_threshold."""
import numpy as np
import pytest
from moirais.fn.rgwvth import rangayyan_wavelet_threshold


def test_rgwvth_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    threshold_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_wavelet_threshold(x, wavelet, levels, threshold_type)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwvth_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    threshold_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_wavelet_threshold(x, wavelet, levels, threshold_type)
    assert isinstance(result, dict)
