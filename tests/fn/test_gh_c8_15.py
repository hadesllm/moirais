"""Tests for gh_c8_15.ghosal_alpha_pst_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c8_15 import ghosal_alpha_pst_crt


def test_gh_c8_15_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_alpha_pst_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_15_edge():
    """Test edge cases."""
    result = ghosal_alpha_pst_crt(np.array([42.0]))
    assert result['n'] == 1
