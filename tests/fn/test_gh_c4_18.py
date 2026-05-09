"""Tests for gh_c4_18.ghosal_dp_mean_dist."""
import numpy as np
import pytest
from moirais.fn.gh_c4_18 import ghosal_dp_mean_dist


def test_gh_c4_18_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_mean_dist(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_18_edge():
    """Test edge cases."""
    result = ghosal_dp_mean_dist(np.array([42.0]))
    assert result['n'] == 1
