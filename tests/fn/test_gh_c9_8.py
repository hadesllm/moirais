"""Tests for gh_c9_8.ghosal_nlar_crt."""
import numpy as np
import pytest
from morie.fn.gh_c9_8 import ghosal_nlar_crt


def test_gh_c9_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_nlar_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c9_8_edge():
    """Test edge cases."""
    result = ghosal_nlar_crt(np.array([42.0]))
    assert result['n'] == 1
