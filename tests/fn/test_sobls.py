"""Tests for sobls.sobol_sequence."""
import numpy as np
import pytest
from moirais.fn.sobls import sobol_sequence


def test_sobls_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sobol_sequence(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_sobls_edge():
    """Test edge cases."""
    result = sobol_sequence(np.array([42.0]))
    assert result['n'] == 1
