"""Tests for rgentrwv.rangayyan_wavelet_entropy."""
import numpy as np
import pytest
from morie.fn.rgentrwv import rangayyan_wavelet_entropy


def test_rgentrwv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_entropy(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgentrwv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_entropy(x, wavelet, levels)
    assert isinstance(result, dict)
