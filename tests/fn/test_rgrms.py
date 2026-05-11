"""Tests for rgrms.rangayyan_rms."""
import numpy as np
import pytest
from morie.fn.rgrms import rangayyan_rms


def test_rgrms_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_rms(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgrms_edge():
    """Test edge cases."""
    result = rangayyan_rms(np.array([42.0]))
    assert result['n'] == 1
