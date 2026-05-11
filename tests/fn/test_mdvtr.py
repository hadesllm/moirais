"""Tests for mdvtr.median_voter."""
import numpy as np
import pytest
from morie.fn.mdvtr import median_voter


def test_mdvtr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = median_voter(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mdvtr_edge():
    """Test edge cases."""
    result = median_voter(np.array([42.0]))
    assert result['n'] == 1
