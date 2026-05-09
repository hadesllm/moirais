"""Tests for gh_c12_10.ghosal_wn_lin_bvm."""
import numpy as np
import pytest
from moirais.fn.gh_c12_10 import ghosal_wn_lin_bvm


def test_gh_c12_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wn_lin_bvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_10_edge():
    """Test edge cases."""
    result = ghosal_wn_lin_bvm(np.array([42.0]))
    assert result['n'] == 1
