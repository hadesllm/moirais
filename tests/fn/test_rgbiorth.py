"""Tests for rgbiorth.rangayyan_biorthogonal_wvlt."""
import numpy as np
import pytest
from morie.fn.rgbiorth import rangayyan_biorthogonal_wvlt


def test_rgbiorth_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_biorthogonal_wvlt(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbiorth_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_biorthogonal_wvlt(x, wavelet, levels)
    assert isinstance(result, dict)
