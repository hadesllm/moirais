"""Tests for gh_c13_14.ghosal_cox_post."""
import numpy as np
import pytest
from morie.fn.gh_c13_14 import ghosal_cox_post


def test_gh_c13_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_post(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_14_edge():
    """Test edge cases."""
    result = ghosal_cox_post(np.array([42.0]))
    assert result['n'] == 1
