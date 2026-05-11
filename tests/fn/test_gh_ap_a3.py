"""Tests for gh_ap_a3.ghosal_tv_distance."""
import numpy as np
import pytest
from morie.fn.gh_ap_a3 import ghosal_tv_distance


def test_gh_ap_a3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_tv_distance(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_a3_edge():
    """Test edge cases."""
    result = ghosal_tv_distance(np.array([42.0]))
    assert result['n'] == 1
