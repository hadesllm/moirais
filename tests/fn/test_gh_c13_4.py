"""Tests for gh_c13_4.ghosal_bp_discrete."""
import numpy as np
import pytest
from morie.fn.gh_c13_4 import ghosal_bp_discrete


def test_gh_c13_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bp_discrete(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_4_edge():
    """Test edge cases."""
    result = ghosal_bp_discrete(np.array([42.0]))
    assert result['n'] == 1
