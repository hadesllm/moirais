"""Tests for gh_c3_7.ghosal_rect_partition."""
import numpy as np
import pytest
from morie.fn.gh_c3_7 import ghosal_rect_partition


def test_gh_c3_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_rect_partition(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c3_7_edge():
    """Test edge cases."""
    result = ghosal_rect_partition(np.array([42.0]))
    assert result['n'] == 1
