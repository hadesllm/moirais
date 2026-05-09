"""Tests for gh_c1_3.ghosal_prior_posterior_update."""
import numpy as np
import pytest
from moirais.fn.gh_c1_3 import ghosal_prior_posterior_update


def test_gh_c1_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_prior_posterior_update(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c1_3_edge():
    """Test edge cases."""
    result = ghosal_prior_posterior_update(np.array([42.0]))
    assert result['n'] == 1
