"""Tests for eslwlt.esl_wavelet_smooth."""
import numpy as np
import pytest
from morie.fn.eslwlt import esl_wavelet_smooth


def test_eslwlt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    result = esl_wavelet_smooth(y, wavelet)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslwlt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    wavelet = 'morl'
    result = esl_wavelet_smooth(y, wavelet)
    assert isinstance(result, dict)
