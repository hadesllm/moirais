"""Tests for gh_c14_9.ghosal_py_process."""
import numpy as np
import pytest
from moirais.fn.gh_c14_9 import ghosal_py_process


def test_gh_c14_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_py_process(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_9_edge():
    """Test edge cases."""
    result = ghosal_py_process(np.array([42.0]))
    assert result['n'] == 1
