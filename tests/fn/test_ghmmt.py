"""Tests for ghmmt.ghosal_moment_matching."""
import numpy as np
import pytest
from moirais.fn.ghmmt import ghosal_moment_matching


def test_ghmmt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_moment_matching(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghmmt_edge():
    """Test edge cases."""
    result = ghosal_moment_matching(np.array([42.0]))
    assert result['n'] == 1
