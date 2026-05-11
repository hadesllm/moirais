"""Tests for gh_ap_c3.ghosal_bracket_num."""
import numpy as np
import pytest
from morie.fn.gh_ap_c3 import ghosal_bracket_num


def test_gh_ap_c3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bracket_num(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_c3_edge():
    """Test edge cases."""
    result = ghosal_bracket_num(np.array([42.0]))
    assert result['n'] == 1
