"""Tests for gh_c1_2.ghosal_absolute_continuity."""
import numpy as np
import pytest
from morie.fn.gh_c1_2 import ghosal_absolute_continuity


def test_gh_c1_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_absolute_continuity(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c1_2_edge():
    """Test edge cases."""
    result = ghosal_absolute_continuity(np.array([42.0]))
    assert result['n'] == 1
