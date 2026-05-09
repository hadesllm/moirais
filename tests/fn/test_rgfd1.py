"""Tests for rgfd1.rangayyan_first_diff."""
import numpy as np
import pytest
from moirais.fn.rgfd1 import rangayyan_first_diff


def test_rgfd1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_first_diff(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgfd1_edge():
    """Test edge cases."""
    result = rangayyan_first_diff(np.array([42.0]))
    assert result['n'] == 1
