"""Tests for fzbrd.fauzi_bias_reduced_kdfe."""
import numpy as np
import pytest
from morie.fn.fzbrd import fauzi_bias_reduced_kdfe


def test_fzbrd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_bias_reduced_kdfe(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzbrd_edge():
    """Test edge cases."""
    result = fauzi_bias_reduced_kdfe(np.array([42.0]))
    assert result['n'] == 1
