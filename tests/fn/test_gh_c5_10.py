"""Tests for gh_c5_10.ghosal_poi_ker."""
import numpy as np
import pytest
from morie.fn.gh_c5_10 import ghosal_poi_ker


def test_gh_c5_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_poi_ker(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_10_edge():
    """Test edge cases."""
    result = ghosal_poi_ker(np.array([42.0]))
    assert result['n'] == 1
