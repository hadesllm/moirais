"""Tests for gh_c4_23.ghosal_pen_dp."""
import numpy as np
import pytest
from morie.fn.gh_c4_23 import ghosal_pen_dp


def test_gh_c4_23_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pen_dp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_23_edge():
    """Test edge cases."""
    result = ghosal_pen_dp(np.array([42.0]))
    assert result['n'] == 1
