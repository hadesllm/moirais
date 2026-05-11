"""Tests for quntf.quantile_function."""
import numpy as np
import pytest
from morie.fn.quntf import quantile_function


def test_quntf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = quantile_function(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_quntf_edge():
    """Test edge cases."""
    result = quantile_function(np.array([42.0]))
    assert result['n'] == 1
