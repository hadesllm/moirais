"""Tests for gh_c8_2.ghosal_ggv_thm."""
import numpy as np
import pytest
from morie.fn.gh_c8_2 import ghosal_ggv_thm


def test_gh_c8_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ggv_thm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_2_edge():
    """Test edge cases."""
    result = ghosal_ggv_thm(np.array([42.0]))
    assert result['n'] == 1
