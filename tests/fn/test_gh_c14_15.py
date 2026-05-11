"""Tests for gh_c14_15.ghosal_ncrm_def."""
import numpy as np
import pytest
from morie.fn.gh_c14_15 import ghosal_ncrm_def


def test_gh_c14_15_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ncrm_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_15_edge():
    """Test edge cases."""
    result = ghosal_ncrm_def(np.array([42.0]))
    assert result['n'] == 1
