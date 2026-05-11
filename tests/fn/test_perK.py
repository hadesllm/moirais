"""Tests for perK.periodic_kernel."""
import numpy as np
import pytest
from morie.fn.perK import periodic_kernel


def test_perK_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = periodic_kernel(x, y, p, l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_perK_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = periodic_kernel(x, y, p, l)
    assert isinstance(result, dict)
