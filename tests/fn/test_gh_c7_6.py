"""Tests for gh_c7_6.ghosal_pt_dens_con."""
import numpy as np
import pytest
from moirais.fn.gh_c7_6 import ghosal_pt_dens_con


def test_gh_c7_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_dens_con(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_6_edge():
    """Test edge cases."""
    result = ghosal_pt_dens_con(np.array([42.0]))
    assert result['n'] == 1
