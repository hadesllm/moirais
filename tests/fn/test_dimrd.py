"""Tests for dimrd.dimensionality_test."""
import numpy as np
import pytest
from moirais.fn.dimrd import dimensionality_test


def test_dimrd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = dimensionality_test(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_dimrd_edge():
    """Test edge cases."""
    result = dimensionality_test(np.array([1.0]))
    assert result['n'] == 1
