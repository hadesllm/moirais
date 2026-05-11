"""Tests for gh_c3_10.ghosal_norm_crm."""
import numpy as np
import pytest
from morie.fn.gh_c3_10 import ghosal_norm_crm


def test_gh_c3_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_norm_crm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c3_10_edge():
    """Test edge cases."""
    result = ghosal_norm_crm(np.array([42.0]))
    assert result['n'] == 1
