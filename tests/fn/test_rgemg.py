"""Tests for rgemg.rangayyan_emg_rms."""
import numpy as np
import pytest
from morie.fn.rgemg import rangayyan_emg_rms


def test_rgemg_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_emg_rms(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgemg_edge():
    """Test edge cases."""
    result = rangayyan_emg_rms(np.array([42.0]))
    assert result['n'] == 1
