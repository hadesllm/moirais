"""Tests for gh_ap_b3.ghosal_renyi_div."""
import numpy as np
import pytest
from morie.fn.gh_ap_b3 import ghosal_renyi_div


def test_gh_ap_b3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_renyi_div(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_b3_edge():
    """Test edge cases."""
    result = ghosal_renyi_div(np.array([42.0]))
    assert result['n'] == 1
