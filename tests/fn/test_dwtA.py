"""Tests for dwtA.discrete_wavelet_anomaly."""
import numpy as np
import pytest
from morie.fn.dwtA import discrete_wavelet_anomaly


def test_dwtA_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = discrete_wavelet_anomaly(x, wavelet, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dwtA_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = 'morl'
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = discrete_wavelet_anomaly(x, wavelet, threshold)
    assert isinstance(result, dict)
