"""Tests for ordct.ordered_categories."""
import numpy as np
import pytest
from morie.fn.ordct import ordered_categories


def test_ordct_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ordered_categories(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ordct_edge():
    """Test edge cases."""
    result = ordered_categories(np.array([42.0]))
    assert result['n'] == 1
