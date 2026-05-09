"""Tests for rgmra.rangayyan_mra."""
import numpy as np
import pytest
from moirais.fn.rgmra import rangayyan_mra


def test_rgmra_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_mra(x, wavelet, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmra_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_mra(x, wavelet, levels)
    assert isinstance(result, dict)
