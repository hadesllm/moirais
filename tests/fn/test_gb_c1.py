"""Tests for gb_c1.gibbons_chebyshev."""
import numpy as np
import pytest
from moirais.fn.gb_c1 import gibbons_chebyshev


def test_gb_c1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_chebyshev(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_c1_edge():
    """Test edge cases."""
    result = gibbons_chebyshev(np.array([42.0]))
    assert result['n'] == 1
