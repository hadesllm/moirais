"""Tests for rotrp.rotary_position_embedding."""
import numpy as np
import pytest
from moirais.fn.rotrp import rotary_position_embedding


def test_rotrp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rotary_position_embedding(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rotrp_edge():
    """Test edge cases."""
    result = rotary_position_embedding(np.array([42.0]))
    assert result['n'] == 1
