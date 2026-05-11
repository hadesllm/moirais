"""Tests for unitl.unit_length_normalization."""
import numpy as np
import pytest
from morie.fn.unitl import unit_length_normalization


def test_unitl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = unit_length_normalization(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_unitl_edge():
    """Test edge cases."""
    result = unit_length_normalization(np.array([42.0]))
    assert result['n'] == 1
