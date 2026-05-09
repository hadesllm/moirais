"""Tests for gh_c14_23.ghosal_ibp_def."""
import numpy as np
import pytest
from moirais.fn.gh_c14_23 import ghosal_ibp_def


def test_gh_c14_23_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ibp_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_23_edge():
    """Test edge cases."""
    result = ghosal_ibp_def(np.array([42.0]))
    assert result['n'] == 1
