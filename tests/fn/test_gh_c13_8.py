"""Tests for gh_c13_8.ghosal_ntr_def."""
import numpy as np
import pytest
from morie.fn.gh_c13_8 import ghosal_ntr_def


def test_gh_c13_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ntr_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_8_edge():
    """Test edge cases."""
    result = ghosal_ntr_def(np.array([42.0]))
    assert result['n'] == 1
