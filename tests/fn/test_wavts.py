"""Tests for wavts.wavelet_time_series."""
import numpy as np
import pytest
from morie.fn.wavts import wavelet_time_series


def test_wavts_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = wavelet_time_series(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_wavts_edge():
    """Test edge cases."""
    result = wavelet_time_series(np.array([42.0]))
    assert result['n'] == 1
