"""Tests for gh_hier_np.ghosal_hierarchical_np."""
import numpy as np
import pytest
from morie.fn.gh_hier_np import ghosal_hierarchical_np


def test_gh_hier_np_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_hierarchical_np(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_hier_np_edge():
    """Test edge cases."""
    result = ghosal_hierarchical_np(np.array([42.0]))
    assert result['n'] == 1
