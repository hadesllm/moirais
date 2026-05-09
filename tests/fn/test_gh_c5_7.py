"""Tests for gh_c5_7.ghosal_pred_rec."""
import numpy as np
import pytest
from moirais.fn.gh_c5_7 import ghosal_pred_rec


def test_gh_c5_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pred_rec(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_7_edge():
    """Test edge cases."""
    result = ghosal_pred_rec(np.array([42.0]))
    assert result['n'] == 1
