"""Tests for foldp.folding_problem."""
import numpy as np
import pytest
from morie.fn.foldp import folding_problem


def test_foldp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = folding_problem(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_foldp_edge():
    """Test edge cases."""
    result = folding_problem(np.array([42.0]))
    assert result['n'] == 1
