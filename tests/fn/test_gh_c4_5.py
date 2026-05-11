"""Tests for gh_c4_5.ghosal_dp_selfsim."""
import numpy as np
import pytest
from morie.fn.gh_c4_5 import ghosal_dp_selfsim


def test_gh_c4_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_selfsim(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_5_edge():
    """Test edge cases."""
    result = ghosal_dp_selfsim(np.array([42.0]))
    assert result['n'] == 1
