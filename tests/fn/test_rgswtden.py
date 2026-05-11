"""Tests for rgswtden.rangayyan_swt_denoise."""
import numpy as np
import pytest
from morie.fn.rgswtden import rangayyan_swt_denoise


def test_rgswtden_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_swt_denoise(x, wavelet, levels, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgswtden_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_swt_denoise(x, wavelet, levels, threshold)
    assert isinstance(result, dict)
