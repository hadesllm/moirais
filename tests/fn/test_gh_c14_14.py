"""Tests for gh_c14_14.ghosal_nig_proc."""
import numpy as np
import pytest
from moirais.fn.gh_c14_14 import ghosal_nig_proc


def test_gh_c14_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_nig_proc(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_14_edge():
    """Test edge cases."""
    result = ghosal_nig_proc(np.array([42.0]))
    assert result['n'] == 1
