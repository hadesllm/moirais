"""Tests for gh_c14_11.ghosal_py_powerlaw."""
import numpy as np
import pytest
from moirais.fn.gh_c14_11 import ghosal_py_powerlaw


def test_gh_c14_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_py_powerlaw(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_11_edge():
    """Test edge cases."""
    result = ghosal_py_powerlaw(np.array([42.0]))
    assert result['n'] == 1
