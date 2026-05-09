"""Tests for gh_c4_22.ghosal_constr_dp."""
import numpy as np
import pytest
from moirais.fn.gh_c4_22 import ghosal_constr_dp


def test_gh_c4_22_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_constr_dp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_22_edge():
    """Test edge cases."""
    result = ghosal_constr_dp(np.array([42.0]))
    assert result['n'] == 1
