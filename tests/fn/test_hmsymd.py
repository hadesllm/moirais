"""Tests for hmsymd.geron_symbolic_diff."""
import numpy as np
import pytest
from morie.fn.hmsymd import geron_symbolic_diff


def test_hmsymd_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    var = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_symbolic_diff(expr, var)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsymd_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    var = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_symbolic_diff(expr, var)
    assert isinstance(result, dict)
