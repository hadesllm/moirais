"""Tests for gh_c11_9.ghosal_statgp_spec."""
import numpy as np
import pytest
from moirais.fn.gh_c11_9 import ghosal_statgp_spec


def test_gh_c11_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_statgp_spec(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c11_9_edge():
    """Test edge cases."""
    result = ghosal_statgp_spec(np.array([42.0]))
    assert result['n'] == 1
