"""Tests for gh_c4_16.ghosal_dp_tails."""
import numpy as np
import pytest
from morie.fn.gh_c4_16 import ghosal_dp_tails


def test_gh_c4_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_tails(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_16_edge():
    """Test edge cases."""
    result = ghosal_dp_tails(np.array([42.0]))
    assert result['n'] == 1
