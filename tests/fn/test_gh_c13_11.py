"""Tests for gh_c13_11.ghosal_ntr_bvm."""
import numpy as np
import pytest
from moirais.fn.gh_c13_11 import ghosal_ntr_bvm


def test_gh_c13_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ntr_bvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_11_edge():
    """Test edge cases."""
    result = ghosal_ntr_bvm(np.array([42.0]))
    assert result['n'] == 1
