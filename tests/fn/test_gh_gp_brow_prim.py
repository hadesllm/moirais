"""Tests for gh_gp_brow_prim.ghosal_gp_brownian_primitive."""
import numpy as np
import pytest
from morie.fn.gh_gp_brow_prim import ghosal_gp_brownian_primitive


def test_gh_gp_brow_prim_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_brownian_primitive(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_gp_brow_prim_edge():
    """Test edge cases."""
    result = ghosal_gp_brownian_primitive(np.array([42.0]))
    assert result['n'] == 1
