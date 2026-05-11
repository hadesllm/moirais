"""Tests for gh_c13_7.ghosal_mix_bp."""
import numpy as np
import pytest
from morie.fn.gh_c13_7 import ghosal_mix_bp


def test_gh_c13_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mix_bp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_7_edge():
    """Test edge cases."""
    result = ghosal_mix_bp(np.array([42.0]))
    assert result['n'] == 1
