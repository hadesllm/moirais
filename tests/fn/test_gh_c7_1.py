"""Tests for gh_c7_1.ghosal_pt_kl_prop."""
import numpy as np
import pytest
from moirais.fn.gh_c7_1 import ghosal_pt_kl_prop


def test_gh_c7_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_kl_prop(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_1_edge():
    """Test edge cases."""
    result = ghosal_pt_kl_prop(np.array([42.0]))
    assert result['n'] == 1
