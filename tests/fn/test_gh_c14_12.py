"""Tests for gh_c14_12.ghosal_pk_process."""
import numpy as np
import pytest
from morie.fn.gh_c14_12 import ghosal_pk_process


def test_gh_c14_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pk_process(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_12_edge():
    """Test edge cases."""
    result = ghosal_pk_process(np.array([42.0]))
    assert result['n'] == 1
