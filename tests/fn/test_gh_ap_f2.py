"""Tests for gh_ap_f2.ghosal_glivenko."""
import numpy as np
import pytest
from morie.fn.gh_ap_f2 import ghosal_glivenko


def test_gh_ap_f2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_glivenko(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_f2_edge():
    """Test edge cases."""
    result = ghosal_glivenko(np.array([42.0]))
    assert result['n'] == 1
