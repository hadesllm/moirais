"""Tests for gh_ap_m1.ghosal_mh_sampler."""
import numpy as np
import pytest
from morie.fn.gh_ap_m1 import ghosal_mh_sampler


def test_gh_ap_m1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mh_sampler(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_m1_edge():
    """Test edge cases."""
    result = ghosal_mh_sampler(np.array([42.0]))
    assert result['n'] == 1
