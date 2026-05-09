"""Tests for gh_ap_i1.ghosal_gp_sample_cont."""
import numpy as np
import pytest
from moirais.fn.gh_ap_i1 import ghosal_gp_sample_cont


def test_gh_ap_i1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_sample_cont(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_i1_edge():
    """Test edge cases."""
    result = ghosal_gp_sample_cont(np.array([42.0]))
    assert result['n'] == 1
