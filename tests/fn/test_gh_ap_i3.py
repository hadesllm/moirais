"""Tests for gh_ap_i3.ghosal_borell_tis."""
import numpy as np
import pytest
from morie.fn.gh_ap_i3 import ghosal_borell_tis


def test_gh_ap_i3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_borell_tis(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_i3_edge():
    """Test edge cases."""
    result = ghosal_borell_tis(np.array([42.0]))
    assert result['n'] == 1
