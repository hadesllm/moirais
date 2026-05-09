"""Tests for gh_c11_2.ghosal_rkhs_norm."""
import numpy as np
import pytest
from moirais.fn.gh_c11_2 import ghosal_rkhs_norm


def test_gh_c11_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_rkhs_norm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_2_edge():
    """Test edge cases."""
    result = ghosal_rkhs_norm(np.array([42.0]))
    assert result['n'] == 1
