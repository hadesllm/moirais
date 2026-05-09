"""Tests for gh_c5_3.ghosal_cgibbs."""
import numpy as np
import pytest
from moirais.fn.gh_c5_3 import ghosal_cgibbs


def test_gh_c5_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cgibbs(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_3_edge():
    """Test edge cases."""
    result = ghosal_cgibbs(np.array([42.0]))
    assert result['n'] == 1
