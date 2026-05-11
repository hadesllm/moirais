"""Tests for gh_c11_8.ghosal_fbm_prior."""
import numpy as np
import pytest
from morie.fn.gh_c11_8 import ghosal_fbm_prior


def test_gh_c11_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_fbm_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_8_edge():
    """Test edge cases."""
    result = ghosal_fbm_prior(np.array([42.0]))
    assert result['n'] == 1
