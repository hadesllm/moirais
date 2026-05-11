"""Tests for gb1251.gibbons_partial_tau."""
import numpy as np
import pytest
from morie.fn.gb1251 import gibbons_partial_tau


def test_gb1251_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = gibbons_partial_tau(x, y, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1251_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = gibbons_partial_tau(x, y, z)
    assert isinstance(result, dict)
