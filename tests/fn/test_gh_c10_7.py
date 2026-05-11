"""Tests for gh_c10_7.ghosal_frs_density."""
import numpy as np
import pytest
from morie.fn.gh_c10_7 import ghosal_frs_density


def test_gh_c10_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_frs_density(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_7_edge():
    """Test edge cases."""
    result = ghosal_frs_density(np.array([42.0]))
    assert result['n'] == 1
