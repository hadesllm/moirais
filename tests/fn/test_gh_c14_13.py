"""Tests for gh_c14_13.ghosal_pk_levy."""
import numpy as np
import pytest
from moirais.fn.gh_c14_13 import ghosal_pk_levy


def test_gh_c14_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pk_levy(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_13_edge():
    """Test edge cases."""
    result = ghosal_pk_levy(np.array([42.0]))
    assert result['n'] == 1
