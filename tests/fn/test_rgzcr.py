"""Tests for rgzcr.rangayyan_zero_crossing."""
import numpy as np
import pytest
from moirais.fn.rgzcr import rangayyan_zero_crossing


def test_rgzcr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_zero_crossing(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgzcr_edge():
    """Test edge cases."""
    result = rangayyan_zero_crossing(np.array([42.0]))
    assert result['n'] == 1
