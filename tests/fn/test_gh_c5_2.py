"""Tests for gh_c5_2.ghosal_dpm_marg."""
import numpy as np
import pytest
from morie.fn.gh_c5_2 import ghosal_dpm_marg


def test_gh_c5_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dpm_marg(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_2_edge():
    """Test edge cases."""
    result = ghosal_dpm_marg(np.array([42.0]))
    assert result['n'] == 1
