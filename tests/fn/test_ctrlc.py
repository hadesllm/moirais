"""Tests for ctrlc.control_comparison."""
import numpy as np
import pytest
from moirais.fn.ctrlc import control_comparison


def test_ctrlc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = control_comparison(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ctrlc_edge():
    """Test edge cases."""
    result = control_comparison(np.array([42.0]))
    assert result['n'] == 1
