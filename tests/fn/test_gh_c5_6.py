"""Tests for gh_c5_6.ghosal_vb_dpm."""
import numpy as np
import pytest
from moirais.fn.gh_c5_6 import ghosal_vb_dpm


def test_gh_c5_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_vb_dpm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_6_edge():
    """Test edge cases."""
    result = ghosal_vb_dpm(np.array([42.0]))
    assert result['n'] == 1
