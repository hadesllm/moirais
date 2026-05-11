"""Tests for cntgc.contingency_coefficient."""
import numpy as np
import pytest
from morie.fn.cntgc import contingency_coefficient


def test_cntgc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = contingency_coefficient(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_cntgc_edge():
    """Test edge cases."""
    result = contingency_coefficient(np.array([42.0]))
    assert result['n'] == 1
