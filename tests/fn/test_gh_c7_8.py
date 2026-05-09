"""Tests for gh_c7_8.ghosal_loc_semipara."""
import numpy as np
import pytest
from moirais.fn.gh_c7_8 import ghosal_loc_semipara


def test_gh_c7_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_loc_semipara(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_8_edge():
    """Test edge cases."""
    result = ghosal_loc_semipara(np.array([42.0]))
    assert result['n'] == 1
