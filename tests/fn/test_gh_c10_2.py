"""Tests for gh_c10_2.ghosal_univ_weights."""
import numpy as np
import pytest
from morie.fn.gh_c10_2 import ghosal_univ_weights


def test_gh_c10_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_univ_weights(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_2_edge():
    """Test edge cases."""
    result = ghosal_univ_weights(np.array([42.0]))
    assert result['n'] == 1
