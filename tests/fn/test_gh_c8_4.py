"""Tests for gh_c8_4.ghosal_prior_mass_cnd."""
import numpy as np
import pytest
from moirais.fn.gh_c8_4 import ghosal_prior_mass_cnd


def test_gh_c8_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_prior_mass_cnd(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_4_edge():
    """Test edge cases."""
    result = ghosal_prior_mass_cnd(np.array([42.0]))
    assert result['n'] == 1
