"""Tests for gh_c11_13.ghosal_gp_adapt_thm."""
import numpy as np
import pytest
from morie.fn.gh_c11_13 import ghosal_gp_adapt_thm


def test_gh_c11_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_adapt_thm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_13_edge():
    """Test edge cases."""
    result = ghosal_gp_adapt_thm(np.array([42.0]))
    assert result['n'] == 1
