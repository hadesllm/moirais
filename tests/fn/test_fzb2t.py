"""Tests for fzb2t.fauzi_b2_coefficient_mrl."""
import numpy as np
import pytest
from morie.fn.fzb2t import fauzi_b2_coefficient_mrl


def test_fzb2t_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b2_coefficient_mrl(t, g_func, density)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb2t_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b2_coefficient_mrl(t, g_func, density)
    assert isinstance(result, dict)
