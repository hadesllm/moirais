"""Tests for gh_c2_5.ghosal_histogram_prior."""
import numpy as np
import pytest
from morie.fn.gh_c2_5 import ghosal_histogram_prior


def test_gh_c2_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_histogram_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c2_5_edge():
    """Test edge cases."""
    result = ghosal_histogram_prior(np.array([42.0]))
    assert result['n'] == 1
