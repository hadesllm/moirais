"""Tests for gh_ap_h1.ghosal_inv_gauss."""
import numpy as np
import pytest
from morie.fn.gh_ap_h1 import ghosal_inv_gauss


def test_gh_ap_h1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_inv_gauss(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_h1_edge():
    """Test edge cases."""
    result = ghosal_inv_gauss(np.array([42.0]))
    assert result['n'] == 1
