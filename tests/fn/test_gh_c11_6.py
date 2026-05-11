"""Tests for gh_c11_6.ghosal_bm_prior."""
import numpy as np
import pytest
from morie.fn.gh_c11_6 import ghosal_bm_prior


def test_gh_c11_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bm_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_6_edge():
    """Test edge cases."""
    result = ghosal_bm_prior(np.array([42.0]))
    assert result['n'] == 1
