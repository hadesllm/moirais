"""Tests for gh_ppt_consist.ghosal_polya_tree_consist_rate."""
import numpy as np
import pytest
from morie.fn.gh_ppt_consist import ghosal_polya_tree_consist_rate


def test_gh_ppt_consist_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_polya_tree_consist_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ppt_consist_edge():
    """Test edge cases."""
    result = ghosal_polya_tree_consist_rate(np.array([42.0]))
    assert result['n'] == 1
