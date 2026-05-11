"""Tests for rgeegrhm.rangayyan_eeg_rhythm_detect."""
import numpy as np
import pytest
from morie.fn.rgeegrhm import rangayyan_eeg_rhythm_detect


def test_rgeegrhm_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_eeg_rhythm_detect(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_rgeegrhm_edge():
    """Test edge cases."""
    result = rangayyan_eeg_rhythm_detect(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
