"""Tests for gb232.gibbons_glivenko_cantelli."""
import numpy as np
import pytest
from moirais.fn.gb232 import gibbons_glivenko_cantelli


def test_gb232_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_glivenko_cantelli(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb232_edge():
    """Test edge cases."""
    result = gibbons_glivenko_cantelli(np.array([42.0]))
    assert result['n'] == 1
