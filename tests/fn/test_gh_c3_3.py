"""Tests for gh_c3_3.ghosal_dir_simplex."""
import numpy as np
import pytest
from morie.fn.gh_c3_3 import ghosal_dir_simplex


def test_gh_c3_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_simplex(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c3_3_edge():
    """Test edge cases."""
    result = ghosal_dir_simplex(np.array([42.0]))
    assert result['n'] == 1
