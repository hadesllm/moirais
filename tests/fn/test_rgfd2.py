"""Tests for rgfd2.rangayyan_second_diff."""
import numpy as np
import pytest
from morie.fn.rgfd2 import rangayyan_second_diff


def test_rgfd2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_second_diff(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgfd2_edge():
    """Test edge cases."""
    result = rangayyan_second_diff(np.array([42.0]))
    assert result['n'] == 1
