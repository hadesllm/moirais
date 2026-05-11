"""Tests for gh_c14_21.ghosal_ord_dep_sbp."""
import numpy as np
import pytest
from morie.fn.gh_c14_21 import ghosal_ord_dep_sbp


def test_gh_c14_21_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ord_dep_sbp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_21_edge():
    """Test edge cases."""
    result = ghosal_ord_dep_sbp(np.array([42.0]))
    assert result['n'] == 1
