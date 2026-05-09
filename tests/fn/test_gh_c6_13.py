"""Tests for gh_c6_13.ghosal_lecam_consist."""
import numpy as np
import pytest
from moirais.fn.gh_c6_13 import ghosal_lecam_consist


def test_gh_c6_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_lecam_consist(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_13_edge():
    """Test edge cases."""
    result = ghosal_lecam_consist(np.array([42.0]))
    assert result['n'] == 1
