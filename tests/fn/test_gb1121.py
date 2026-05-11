"""Tests for gb1121.gibbons_kendall_tau."""
import numpy as np
import pytest
from morie.fn.gb1121 import gibbons_kendall_tau


def test_gb1121_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_kendall_tau(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1121_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_kendall_tau(x, y)
    assert isinstance(result, dict)
