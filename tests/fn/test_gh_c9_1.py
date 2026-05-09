"""Tests for gh_c9_1.ghosal_logspline_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c9_1 import ghosal_logspline_crt


def test_gh_c9_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_logspline_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c9_1_edge():
    """Test edge cases."""
    result = ghosal_logspline_crt(np.array([42.0]))
    assert result['n'] == 1
