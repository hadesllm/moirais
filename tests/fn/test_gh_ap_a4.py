"""Tests for gh_ap_a4.ghosal_hellinger_dist."""
import numpy as np
import pytest
from morie.fn.gh_ap_a4 import ghosal_hellinger_dist


def test_gh_ap_a4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_hellinger_dist(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_a4_edge():
    """Test edge cases."""
    result = ghosal_hellinger_dist(np.array([42.0]))
    assert result['n'] == 1
