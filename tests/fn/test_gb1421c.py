"""Tests for gb1421c.gibbons_contingency_coeff."""
import numpy as np
import pytest
from moirais.fn.gb1421c import gibbons_contingency_coeff


def test_gb1421c_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_contingency_coeff(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb1421c_edge():
    """Test edge cases."""
    result = gibbons_contingency_coeff(np.array([42.0]))
    assert result['n'] == 1
