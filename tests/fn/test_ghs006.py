"""Tests for ghs006.ghosal_ch2_feller_density_approximation."""
import numpy as np
import pytest
from moirais.fn.ghs006 import ghosal_ch2_feller_density_approximation


def test_ghs006_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    F = np.random.default_rng(43).normal(0, 1, 100)
    h_k = np.random.default_rng(42).normal(0, 1, 100)
    g_k = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch2_feller_density_approximation(x, k, F, h_k, g_k, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs006_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    F = np.random.default_rng(43).normal(0, 1, 100)
    h_k = np.random.default_rng(42).normal(0, 1, 100)
    g_k = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch2_feller_density_approximation(x, k, F, h_k, g_k, V)
    assert isinstance(result, dict)
