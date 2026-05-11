"""Tests for gh_c13_12.ghosal_smhaz_gp."""
import numpy as np
import pytest
from morie.fn.gh_c13_12 import ghosal_smhaz_gp


def test_gh_c13_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_smhaz_gp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_12_edge():
    """Test edge cases."""
    result = ghosal_smhaz_gp(np.array([42.0]))
    assert result['n'] == 1
