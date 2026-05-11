"""Tests for merck.mercer_theorem."""
import numpy as np
import pytest
from morie.fn.merck import mercer_theorem


def test_merck_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mercer_theorem(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_merck_edge():
    """Test edge cases."""
    result = mercer_theorem(np.array([42.0]))
    assert result['n'] == 1
