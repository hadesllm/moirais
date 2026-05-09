"""Tests for gh_c12_5.ghosal_eff_infl_fn."""
import numpy as np
import pytest
from moirais.fn.gh_c12_5 import ghosal_eff_infl_fn


def test_gh_c12_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_eff_infl_fn(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c12_5_edge():
    """Test edge cases."""
    result = ghosal_eff_infl_fn(np.array([42.0]))
    assert result['n'] == 1
