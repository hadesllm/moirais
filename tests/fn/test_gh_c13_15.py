"""Tests for gh_c13_15.ghosal_cox_bvm."""
import numpy as np
import pytest
from moirais.fn.gh_c13_15 import ghosal_cox_bvm


def test_gh_c13_15_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_bvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_15_edge():
    """Test edge cases."""
    result = ghosal_cox_bvm(np.array([42.0]))
    assert result['n'] == 1
