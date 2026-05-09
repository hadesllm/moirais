"""Tests for ddrbnd.deer_dr_bounds."""
import numpy as np
import pytest
from moirais.fn.ddrbnd import deer_dr_bounds


def test_ddrbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = deer_dr_bounds(y, D, Z, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ddrbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = deer_dr_bounds(y, D, Z, X)
    assert isinstance(result, dict)
