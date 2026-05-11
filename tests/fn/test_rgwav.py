"""Tests for rgwav.rangayyan_wavelet_denoise."""
import numpy as np
import pytest
from morie.fn.rgwav import rangayyan_wavelet_denoise


def test_rgwav_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_wavelet_denoise(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgwav_edge():
    """Test edge cases."""
    result = rangayyan_wavelet_denoise(np.array([42.0]))
    assert result['n'] == 1
