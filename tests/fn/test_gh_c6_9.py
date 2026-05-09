"""Tests for gh_c6_9.ghosal_kl_perm."""
import numpy as np
import pytest
from moirais.fn.gh_c6_9 import ghosal_kl_perm


def test_gh_c6_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_kl_perm(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_9_edge():
    """Test edge cases."""
    result = ghosal_kl_perm(np.array([42.0]))
    assert result['n'] == 1
