"""Tests for rgdwt.rangayyan_dwt."""
import numpy as np
import pytest
from morie.fn.rgdwt import rangayyan_dwt


def test_rgdwt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_dwt(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgdwt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_dwt(x, wavelet, levels)
    assert isinstance(result, dict)
