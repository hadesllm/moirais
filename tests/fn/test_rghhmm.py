"""Tests for rghhmm.rangayyan_hodgkin_huxley."""
import numpy as np
import pytest
from morie.fn.rghhmm import rangayyan_hodgkin_huxley


def test_rghhmm_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    I_ext = np.random.default_rng(42).normal(0, 1, 100)
    g_Na = np.random.default_rng(42).normal(0, 1, 100)
    g_K = np.random.default_rng(42).normal(0, 1, 100)
    g_L = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hodgkin_huxley(t, I_ext, g_Na, g_K, g_L, C_m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghhmm_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    I_ext = np.random.default_rng(42).normal(0, 1, 100)
    g_Na = np.random.default_rng(42).normal(0, 1, 100)
    g_K = np.random.default_rng(42).normal(0, 1, 100)
    g_L = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hodgkin_huxley(t, I_ext, g_Na, g_K, g_L, C_m)
    assert isinstance(result, dict)
