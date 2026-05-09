"""Tests for gh_c6_10.ghosal_non_iid_con."""
import numpy as np
import pytest
from moirais.fn.gh_c6_10 import ghosal_non_iid_con


def test_gh_c6_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_non_iid_con(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_10_edge():
    """Test edge cases."""
    result = ghosal_non_iid_con(np.array([42.0]))
    assert result['n'] == 1
