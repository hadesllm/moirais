"""Tests for gh_c10_6.ghosal_rnd_series_pr."""
import numpy as np
import pytest
from morie.fn.gh_c10_6 import ghosal_rnd_series_pr


def test_gh_c10_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_rnd_series_pr(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_6_edge():
    """Test edge cases."""
    result = ghosal_rnd_series_pr(np.array([42.0]))
    assert result['n'] == 1
