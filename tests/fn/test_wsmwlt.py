"""Tests for wsmwlt.wasserman_wavelet_smooth."""
import numpy as np
import pytest
from moirais.fn.wsmwlt import wasserman_wavelet_smooth


def test_wsmwlt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    sigma = 1.0
    result = wasserman_wavelet_smooth(y, wavelet, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmwlt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    sigma = 1.0
    result = wasserman_wavelet_smooth(y, wavelet, sigma)
    assert isinstance(result, dict)
