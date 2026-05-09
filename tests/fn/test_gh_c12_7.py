"""Tests for gh_c12_7.ghosal_strict_sbvm."""
import numpy as np
import pytest
from moirais.fn.gh_c12_7 import ghosal_strict_sbvm


def test_gh_c12_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_strict_sbvm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_7_edge():
    """Test edge cases."""
    result = ghosal_strict_sbvm(np.array([42.0]))
    assert result['n'] == 1
