"""Tests for rgdft.rangayyan_dft."""
import numpy as np
import pytest
from moirais.fn.rgdft import rangayyan_dft


def test_rgdft_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_dft(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgdft_edge():
    """Test edge cases."""
    result = rangayyan_dft(np.array([42.0]))
    assert result['n'] == 1
