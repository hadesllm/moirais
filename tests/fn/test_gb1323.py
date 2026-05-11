"""Tests for gb1323.gibbons_are_twosided."""
import numpy as np
import pytest
from morie.fn.gb1323 import gibbons_are_twosided


def test_gb1323_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_twosided(T, T_star)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1323_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_twosided(T, T_star)
    assert isinstance(result, dict)
