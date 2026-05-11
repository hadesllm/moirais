"""Tests for fzc2x.fauzi_c2_coefficient."""
import numpy as np
import pytest
from morie.fn.fzc2x import fauzi_c2_coefficient


def test_fzc2x_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_c2_coefficient(x, g_func, density)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzc2x_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_c2_coefficient(x, g_func, density)
    assert isinstance(result, dict)
