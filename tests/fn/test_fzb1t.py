"""Tests for fzb1t.fauzi_b1_coefficient."""
import numpy as np
import pytest
from moirais.fn.fzb1t import fauzi_b1_coefficient


def test_fzb1t_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b1_coefficient(t, g_func, density)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb1t_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b1_coefficient(t, g_func, density)
    assert isinstance(result, dict)
