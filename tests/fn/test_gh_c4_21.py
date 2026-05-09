"""Tests for gh_c4_21.ghosal_inv_dp."""
import numpy as np
import pytest
from moirais.fn.gh_c4_21 import ghosal_inv_dp


def test_gh_c4_21_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_inv_dp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_21_edge():
    """Test edge cases."""
    result = ghosal_inv_dp(np.array([42.0]))
    assert result['n'] == 1
