"""Tests for gh_c7_3.ghosal_exp_dens_kl."""
import numpy as np
import pytest
from morie.fn.gh_c7_3 import ghosal_exp_dens_kl


def test_gh_c7_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_exp_dens_kl(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_3_edge():
    """Test edge cases."""
    result = ghosal_exp_dens_kl(np.array([42.0]))
    assert result['n'] == 1
