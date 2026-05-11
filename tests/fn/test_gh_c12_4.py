"""Tests for gh_c12_4.ghosal_semipara_bvm."""
import numpy as np
import pytest
from morie.fn.gh_c12_4 import ghosal_semipara_bvm


def test_gh_c12_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_semipara_bvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_4_edge():
    """Test edge cases."""
    result = ghosal_semipara_bvm(np.array([42.0]))
    assert result['n'] == 1
