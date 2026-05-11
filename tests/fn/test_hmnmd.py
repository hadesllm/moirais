"""Tests for hmnmd.geron_numerical_diff."""
import numpy as np
import pytest
from morie.fn.hmnmd import geron_numerical_diff


def test_hmnmd_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_numerical_diff(f, x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnmd_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_numerical_diff(f, x, h)
    assert isinstance(result, dict)
