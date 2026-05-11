"""Tests for gh_gp_orn_uhl.ghosal_gp_ornstein_uhlenbeck."""
import numpy as np
import pytest
from morie.fn.gh_gp_orn_uhl import ghosal_gp_ornstein_uhlenbeck


def test_gh_gp_orn_uhl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_ornstein_uhlenbeck(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_gp_orn_uhl_edge():
    """Test edge cases."""
    result = ghosal_gp_ornstein_uhlenbeck(np.array([42.0]))
    assert result['n'] == 1
