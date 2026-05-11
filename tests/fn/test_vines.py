"""Tests for vines.vine_copula."""
import numpy as np
import pytest
from morie.fn.vines import vine_copula


def test_vines_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = vine_copula(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_vines_edge():
    """Test edge cases."""
    result = vine_copula(np.array([42.0]))
    assert result['n'] == 1
