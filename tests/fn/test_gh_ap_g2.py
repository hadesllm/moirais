"""Tests for gh_ap_g2.ghosal_dir_moments."""
import numpy as np
import pytest
from morie.fn.gh_ap_g2 import ghosal_dir_moments


def test_gh_ap_g2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_moments(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_g2_edge():
    """Test edge cases."""
    result = ghosal_dir_moments(np.array([42.0]))
    assert result['n'] == 1
