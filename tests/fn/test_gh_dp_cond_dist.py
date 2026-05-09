"""Tests for gh_dp_cond_dist.ghosal_dp_conditional_distribution."""
import numpy as np
import pytest
from moirais.fn.gh_dp_cond_dist import ghosal_dp_conditional_distribution


def test_gh_dp_cond_dist_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_conditional_distribution(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_dp_cond_dist_edge():
    """Test edge cases."""
    result = ghosal_dp_conditional_distribution(np.array([42.0]))
    assert result['n'] == 1
