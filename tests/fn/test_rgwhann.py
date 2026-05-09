"""Tests for rgwhann.rangayyan_hann_window."""
import numpy as np
import pytest
from moirais.fn.rgwhann import rangayyan_hann_window


def test_rgwhann_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_hann_window(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgwhann_edge():
    """Test edge cases."""
    result = rangayyan_hann_window(np.array([42.0]))
    assert result['n'] == 1
