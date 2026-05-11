"""Tests for gh_c10_4.ghosal_two_model_adp."""
import numpy as np
import pytest
from morie.fn.gh_c10_4 import ghosal_two_model_adp


def test_gh_c10_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_two_model_adp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_4_edge():
    """Test edge cases."""
    result = ghosal_two_model_adp(np.array([42.0]))
    assert result['n'] == 1
