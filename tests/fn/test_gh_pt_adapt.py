"""Tests for gh_pt_adapt.ghosal_pt_adaptive."""
import numpy as np
import pytest
from moirais.fn.gh_pt_adapt import ghosal_pt_adaptive


def test_gh_pt_adapt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_adaptive(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_pt_adapt_edge():
    """Test edge cases."""
    result = ghosal_pt_adaptive(np.array([42.0]))
    assert result['n'] == 1
