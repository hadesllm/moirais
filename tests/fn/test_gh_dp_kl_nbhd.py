"""Tests for gh_dp_kl_nbhd.ghosal_dp_kl_nbhd_mass."""
import numpy as np
import pytest
from morie.fn.gh_dp_kl_nbhd import ghosal_dp_kl_nbhd_mass


def test_gh_dp_kl_nbhd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_kl_nbhd_mass(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_dp_kl_nbhd_edge():
    """Test edge cases."""
    result = ghosal_dp_kl_nbhd_mass(np.array([42.0]))
    assert result['n'] == 1
