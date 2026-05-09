"""Tests for gh_c6_4.ghosal_df_inconsist."""
import numpy as np
import pytest
from moirais.fn.gh_c6_4 import ghosal_df_inconsist


def test_gh_c6_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_df_inconsist(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c6_4_edge():
    """Test edge cases."""
    result = ghosal_df_inconsist(np.array([42.0]))
    assert result['n'] == 1
