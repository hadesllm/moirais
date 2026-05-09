"""Tests for gh_c14_6.ghosal_ssp_post."""
import numpy as np
import pytest
from moirais.fn.gh_c14_6 import ghosal_ssp_post


def test_gh_c14_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ssp_post(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_6_edge():
    """Test edge cases."""
    result = ghosal_ssp_post(np.array([42.0]))
    assert result['n'] == 1
