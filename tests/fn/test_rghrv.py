"""Tests for rghrv.rangayyan_hrv."""
import numpy as np
import pytest
from morie.fn.rghrv import rangayyan_hrv


def test_rghrv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_hrv(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rghrv_edge():
    """Test edge cases."""
    result = rangayyan_hrv(np.array([42.0]))
    assert result['n'] == 1
