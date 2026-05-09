"""Tests for gh_mises_eff.ghosal_mises_efficiency."""
import numpy as np
import pytest
from moirais.fn.gh_mises_eff import ghosal_mises_efficiency


def test_gh_mises_eff_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mises_efficiency(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_mises_eff_edge():
    """Test edge cases."""
    result = ghosal_mises_efficiency(np.array([42.0]))
    assert result['n'] == 1
