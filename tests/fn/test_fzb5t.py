"""Tests for fzb5t.fauzi_b5_coefficient_mrl."""
import numpy as np
import pytest
from moirais.fn.fzb5t import fauzi_b5_coefficient_mrl


def test_fzb5t_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    mrl = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b5_coefficient_mrl(t, g_func, density, mrl)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb5t_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    g_func = (lambda v: v)
    density = np.random.default_rng(42).normal(0, 1, 100)
    mrl = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b5_coefficient_mrl(t, g_func, density, mrl)
    assert isinstance(result, dict)
