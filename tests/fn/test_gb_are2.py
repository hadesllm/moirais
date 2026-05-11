"""Tests for gb_are2.gibbons_are_normal_case."""
import numpy as np
import pytest
from morie.fn.gb_are2 import gibbons_are_normal_case


def test_gb_are2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_are_normal_case(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_are2_edge():
    """Test edge cases."""
    result = gibbons_are_normal_case(np.array([42.0]))
    assert result['n'] == 1
