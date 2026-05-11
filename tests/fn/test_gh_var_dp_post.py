"""Tests for gh_var_dp_post.ghosal_variational_dp_posterior."""
import numpy as np
import pytest
from morie.fn.gh_var_dp_post import ghosal_variational_dp_posterior


def test_gh_var_dp_post_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_variational_dp_posterior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_var_dp_post_edge():
    """Test edge cases."""
    result = ghosal_variational_dp_posterior(np.array([42.0]))
    assert result['n'] == 1
