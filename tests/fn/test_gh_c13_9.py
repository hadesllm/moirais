"""Tests for gh_c13_9.ghosal_ntr_levy."""
import numpy as np
import pytest
from morie.fn.gh_c13_9 import ghosal_ntr_levy


def test_gh_c13_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ntr_levy(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_9_edge():
    """Test edge cases."""
    result = ghosal_ntr_levy(np.array([42.0]))
    assert result['n'] == 1
