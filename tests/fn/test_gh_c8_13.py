"""Tests for gh_c8_13.ghosal_misspec_crt."""
import numpy as np
import pytest
from morie.fn.gh_c8_13 import ghosal_misspec_crt


def test_gh_c8_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_misspec_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_13_edge():
    """Test edge cases."""
    result = ghosal_misspec_crt(np.array([42.0]))
    assert result['n'] == 1
