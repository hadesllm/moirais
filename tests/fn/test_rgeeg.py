"""Tests for rgeeg.rangayyan_eeg_bands."""
import numpy as np
import pytest
from morie.fn.rgeeg import rangayyan_eeg_bands


def test_rgeeg_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_eeg_bands(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgeeg_edge():
    """Test edge cases."""
    result = rangayyan_eeg_bands(np.array([42.0]))
    assert result['n'] == 1
