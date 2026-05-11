"""Tests for gh_c8_6.ghosal_iid_crt_thm."""
import numpy as np
import pytest
from morie.fn.gh_c8_6 import ghosal_iid_crt_thm


def test_gh_c8_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_iid_crt_thm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_6_edge():
    """Test edge cases."""
    result = ghosal_iid_crt_thm(np.array([42.0]))
    assert result['n'] == 1
