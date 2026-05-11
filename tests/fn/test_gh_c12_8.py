"""Tests for gh_c12_8.ghosal_cox_bvm_sp."""
import numpy as np
import pytest
from morie.fn.gh_c12_8 import ghosal_cox_bvm_sp


def test_gh_c12_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_bvm_sp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_8_edge():
    """Test edge cases."""
    result = ghosal_cox_bvm_sp(np.array([42.0]))
    assert result['n'] == 1
