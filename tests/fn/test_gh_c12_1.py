"""Tests for gh_c12_1.ghosal_infdim_bvm."""
import numpy as np
import pytest
from morie.fn.gh_c12_1 import ghosal_infdim_bvm


def test_gh_c12_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_infdim_bvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_1_edge():
    """Test edge cases."""
    result = ghosal_infdim_bvm(np.array([42.0]))
    assert result['n'] == 1
