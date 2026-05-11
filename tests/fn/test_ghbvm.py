"""Tests for ghbvm.ghosal_bernstein_von_mises."""
import numpy as np
import pytest
from morie.fn.ghbvm import ghosal_bernstein_von_mises


def test_ghbvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bernstein_von_mises(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghbvm_edge():
    """Test edge cases."""
    result = ghosal_bernstein_von_mises(np.array([42.0]))
    assert result['n'] == 1
