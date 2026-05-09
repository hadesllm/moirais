"""Tests for gh_c5_5.ghosal_blk_gibbs."""
import numpy as np
import pytest
from moirais.fn.gh_c5_5 import ghosal_blk_gibbs


def test_gh_c5_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_blk_gibbs(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_5_edge():
    """Test edge cases."""
    result = ghosal_blk_gibbs(np.array([42.0]))
    assert result['n'] == 1
