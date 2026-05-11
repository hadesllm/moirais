"""Tests for gh_c11_10.ghosal_series_gp."""
import numpy as np
import pytest
from morie.fn.gh_c11_10 import ghosal_series_gp


def test_gh_c11_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_series_gp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_10_edge():
    """Test edge cases."""
    result = ghosal_series_gp(np.array([42.0]))
    assert result['n'] == 1
