"""Tests for mmaxn.minmax_normalization."""
import numpy as np
import pytest
from moirais.fn.mmaxn import minmax_normalization


def test_mmaxn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = minmax_normalization(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mmaxn_edge():
    """Test edge cases."""
    result = minmax_normalization(np.array([42.0]))
    assert result['n'] == 1
