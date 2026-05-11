"""Tests for gh_ap_m2.ghosal_gibbs_sampler."""
import numpy as np
import pytest
from morie.fn.gh_ap_m2 import ghosal_gibbs_sampler


def test_gh_ap_m2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gibbs_sampler(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_m2_edge():
    """Test edge cases."""
    result = ghosal_gibbs_sampler(np.array([42.0]))
    assert result['n'] == 1
