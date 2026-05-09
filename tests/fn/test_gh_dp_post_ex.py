"""Tests for gh_dp_post_ex.ghosal_dp_posterior_exact."""
import numpy as np
import pytest
from moirais.fn.gh_dp_post_ex import ghosal_dp_posterior_exact


def test_gh_dp_post_ex_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_posterior_exact(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_dp_post_ex_edge():
    """Test edge cases."""
    result = ghosal_dp_posterior_exact(np.array([42.0]))
    assert result['n'] == 1
