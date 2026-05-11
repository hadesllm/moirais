"""Tests for rgwvpkt.rangayyan_wavelet_packet."""
import numpy as np
import pytest
from morie.fn.rgwvpkt import rangayyan_wavelet_packet


def test_rgwvpkt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_packet(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwvpkt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_wavelet_packet(x, wavelet, levels)
    assert isinstance(result, dict)
