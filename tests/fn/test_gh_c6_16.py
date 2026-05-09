"""Tests for gh_c6_16.ghosal_alpha_post."""
import numpy as np
import pytest
from moirais.fn.gh_c6_16 import ghosal_alpha_post


def test_gh_c6_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_alpha_post(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_16_edge():
    """Test edge cases."""
    result = ghosal_alpha_post(np.array([42.0]))
    assert result['n'] == 1
