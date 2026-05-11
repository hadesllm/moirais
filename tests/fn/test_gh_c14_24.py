"""Tests for gh_c14_24.ghosal_ibp_stickbr."""
import numpy as np
import pytest
from morie.fn.gh_c14_24 import ghosal_ibp_stickbr


def test_gh_c14_24_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ibp_stickbr(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_24_edge():
    """Test edge cases."""
    result = ghosal_ibp_stickbr(np.array([42.0]))
    assert result['n'] == 1
