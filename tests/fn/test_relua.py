"""Tests for relua.relu_activation."""
import numpy as np
import pytest
from morie.fn.relua import relu_activation


def test_relua_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = relu_activation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_relua_edge():
    """Test edge cases."""
    result = relu_activation(np.array([42.0]))
    assert result['n'] == 1
