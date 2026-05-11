"""Tests for gh_c10_3.ghosal_param_rate."""
import numpy as np
import pytest
from morie.fn.gh_c10_3 import ghosal_param_rate


def test_gh_c10_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_param_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_3_edge():
    """Test edge cases."""
    result = ghosal_param_rate(np.array([42.0]))
    assert result['n'] == 1
