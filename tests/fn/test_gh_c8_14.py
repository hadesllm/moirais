"""Tests for gh_c8_14.ghosal_convex_misp."""
import numpy as np
import pytest
from moirais.fn.gh_c8_14 import ghosal_convex_misp


def test_gh_c8_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_convex_misp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_14_edge():
    """Test edge cases."""
    result = ghosal_convex_misp(np.array([42.0]))
    assert result['n'] == 1
