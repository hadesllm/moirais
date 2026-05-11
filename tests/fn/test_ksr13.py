"""Tests for ksr13.kosorok_tangent_space."""
import numpy as np
import pytest
from morie.fn.ksr13 import kosorok_tangent_space


def test_ksr13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_tangent_space(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ksr13_edge():
    """Test edge cases."""
    result = kosorok_tangent_space(np.array([42.0]))
    assert result['n'] == 1
