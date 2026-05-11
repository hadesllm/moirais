"""Tests for gh_besov_prior.ghosal_besov_prior."""
import numpy as np
import pytest
from morie.fn.gh_besov_prior import ghosal_besov_prior


def test_gh_besov_prior_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_besov_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_besov_prior_edge():
    """Test edge cases."""
    result = ghosal_besov_prior(np.array([42.0]))
    assert result['n'] == 1
