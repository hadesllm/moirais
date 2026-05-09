"""Tests for gh_c7_7.ghosal_spec_dens_con."""
import numpy as np
import pytest
from moirais.fn.gh_c7_7 import ghosal_spec_dens_con


def test_gh_c7_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_spec_dens_con(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_7_edge():
    """Test edge cases."""
    result = ghosal_spec_dens_con(np.array([42.0]))
    assert result['n'] == 1
