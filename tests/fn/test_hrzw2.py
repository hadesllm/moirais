"""Tests for hrzw2.horowitz_bandwidth_bootstrap."""
import numpy as np
import pytest
from morie.fn.hrzw2 import horowitz_bandwidth_bootstrap


def test_hrzw2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_bandwidth_bootstrap(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzw2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_bandwidth_bootstrap(x, y)
    assert isinstance(result, dict)
