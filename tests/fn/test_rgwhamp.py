"""Tests for rgwhamp.rangayyan_hamming_window."""
import numpy as np
import pytest
from morie.fn.rgwhamp import rangayyan_hamming_window


def test_rgwhamp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_hamming_window(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgwhamp_edge():
    """Test edge cases."""
    result = rangayyan_hamming_window(np.array([42.0]))
    assert result['n'] == 1
