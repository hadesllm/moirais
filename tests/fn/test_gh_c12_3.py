"""Tests for gh_c12_3.ghosal_strong_apx_dp."""
import numpy as np
import pytest
from moirais.fn.gh_c12_3 import ghosal_strong_apx_dp


def test_gh_c12_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_strong_apx_dp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_3_edge():
    """Test edge cases."""
    result = ghosal_strong_apx_dp(np.array([42.0]))
    assert result['n'] == 1
