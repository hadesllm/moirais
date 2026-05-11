"""Tests for gh_c7_4.ghosal_norm_mix_con."""
import numpy as np
import pytest
from morie.fn.gh_c7_4 import ghosal_norm_mix_con


def test_gh_c7_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_norm_mix_con(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_4_edge():
    """Test edge cases."""
    result = ghosal_norm_mix_con(np.array([42.0]))
    assert result['n'] == 1
