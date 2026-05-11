"""Tests for gh_c11_1.ghosal_gp_def_rkhs."""
import numpy as np
import pytest
from morie.fn.gh_c11_1 import ghosal_gp_def_rkhs


def test_gh_c11_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_def_rkhs(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_1_edge():
    """Test edge cases."""
    result = ghosal_gp_def_rkhs(np.array([42.0]))
    assert result['n'] == 1
