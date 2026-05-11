"""Tests for gh_c6_3.ghosal_doob_consist."""
import numpy as np
import pytest
from morie.fn.gh_c6_3 import ghosal_doob_consist


def test_gh_c6_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_doob_consist(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_3_edge():
    """Test edge cases."""
    result = ghosal_doob_consist(np.array([42.0]))
    assert result['n'] == 1
