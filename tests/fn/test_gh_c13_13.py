"""Tests for gh_c13_13.ghosal_cox_model."""
import numpy as np
import pytest
from morie.fn.gh_c13_13 import ghosal_cox_model


def test_gh_c13_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_model(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_13_edge():
    """Test edge cases."""
    result = ghosal_cox_model(np.array([42.0]))
    assert result['n'] == 1
