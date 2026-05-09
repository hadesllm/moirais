"""Tests for gh_iid_consist.ghosal_iid_posterior_consistency."""
import numpy as np
import pytest
from moirais.fn.gh_iid_consist import ghosal_iid_posterior_consistency


def test_gh_iid_consist_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_iid_posterior_consistency(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_iid_consist_edge():
    """Test edge cases."""
    result = ghosal_iid_posterior_consistency(np.array([42.0]))
    assert result['n'] == 1
