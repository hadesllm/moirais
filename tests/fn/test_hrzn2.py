"""Tests for hrzn2.horowitz_deconvolution."""
import numpy as np
import pytest
from morie.fn.hrzn2 import horowitz_deconvolution


def test_hrzn2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = horowitz_deconvolution(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_hrzn2_edge():
    """Test edge cases."""
    result = horowitz_deconvolution(np.array([42.0]))
    assert result['n'] == 1
