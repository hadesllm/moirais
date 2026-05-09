"""Tests for rcall.roll_call_analysis."""
import numpy as np
import pytest
from moirais.fn.rcall import roll_call_analysis


def test_rcall_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = roll_call_analysis(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rcall_edge():
    """Test edge cases."""
    result = roll_call_analysis(np.array([42.0]))
    assert result['n'] == 1
