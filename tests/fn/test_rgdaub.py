"""Tests for rgdaub.rangayyan_daubechies."""
import numpy as np
import pytest
from morie.fn.rgdaub import rangayyan_daubechies


def test_rgdaub_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_daubechies(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgdaub_edge():
    """Test edge cases."""
    result = rangayyan_daubechies(np.array([42.0]))
    assert result['n'] == 1
