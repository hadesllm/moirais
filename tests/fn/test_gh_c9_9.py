"""Tests for gh_c9_9.ghosal_wn_conj_crt."""
import numpy as np
import pytest
from morie.fn.gh_c9_9 import ghosal_wn_conj_crt


def test_gh_c9_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wn_conj_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c9_9_edge():
    """Test edge cases."""
    result = ghosal_wn_conj_crt(np.array([42.0]))
    assert result['n'] == 1
