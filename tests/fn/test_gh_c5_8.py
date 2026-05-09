"""Tests for gh_c5_8.ghosal_gauss_ker."""
import numpy as np
import pytest
from moirais.fn.gh_c5_8 import ghosal_gauss_ker


def test_gh_c5_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gauss_ker(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_8_edge():
    """Test edge cases."""
    result = ghosal_gauss_ker(np.array([42.0]))
    assert result['n'] == 1
