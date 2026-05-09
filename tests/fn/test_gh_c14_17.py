"""Tests for gh_c14_17.ghosal_disc_rp_rel."""
import numpy as np
import pytest
from moirais.fn.gh_c14_17 import ghosal_disc_rp_rel


def test_gh_c14_17_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_disc_rp_rel(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_17_edge():
    """Test edge cases."""
    result = ghosal_disc_rp_rel(np.array([42.0]))
    assert result['n'] == 1
