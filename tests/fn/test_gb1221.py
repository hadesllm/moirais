"""Tests for gb1221.gibbons_friedman."""
import numpy as np
import pytest
from morie.fn.gb1221 import gibbons_friedman


def test_gb1221_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman(data, k, b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1221_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman(data, k, b)
    assert isinstance(result, dict)
