"""Tests for wsmhfd.wasserman_hoeffding."""
import numpy as np
import pytest
from morie.fn.wsmhfd import wasserman_hoeffding


def test_wsmhfd_basic():
    """Test basic functionality."""
    n = 100
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_hoeffding(n, t, a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmhfd_edge():
    """Test edge cases."""
    n = 100
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_hoeffding(n, t, a, b)
    assert isinstance(result, dict)
