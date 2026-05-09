"""Tests for gb1321.gibbons_are_def."""
import numpy as np
import pytest
from moirais.fn.gb1321 import gibbons_are_def


def test_gb1321_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_are_def(T, T_star, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1321_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_are_def(T, T_star, n)
    assert isinstance(result, dict)
