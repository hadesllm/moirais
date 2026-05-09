"""Tests for gh_c13_16.ghosal_bb_censored."""
import numpy as np
import pytest
from moirais.fn.gh_c13_16 import ghosal_bb_censored


def test_gh_c13_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bb_censored(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_16_edge():
    """Test edge cases."""
    result = ghosal_bb_censored(np.array([42.0]))
    assert result['n'] == 1
