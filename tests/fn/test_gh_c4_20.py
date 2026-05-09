"""Tests for gh_c4_20.ghosal_mix_dp."""
import numpy as np
import pytest
from moirais.fn.gh_c4_20 import ghosal_mix_dp


def test_gh_c4_20_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mix_dp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_20_edge():
    """Test edge cases."""
    result = ghosal_mix_dp(np.array([42.0]))
    assert result['n'] == 1
