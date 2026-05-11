"""Tests for unfdl.unfolding_analysis."""
import numpy as np
import pytest
from morie.fn.unfdl import unfolding_analysis


def test_unfdl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = unfolding_analysis(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_unfdl_edge():
    """Test edge cases."""
    result = unfolding_analysis(np.array([42.0]))
    assert result['n'] == 1
