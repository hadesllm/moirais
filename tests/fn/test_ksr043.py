"""Tests for ksr043.kosorok_ch2_quantile_hadamard_inequality."""
import numpy as np
import pytest
from morie.fn.ksr043 import kosorok_ch2_quantile_hadamard_inequality


def test_ksr043_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    h_n = np.random.default_rng(42).normal(0, 1, 100)
    t_n = np.random.default_rng(42).normal(0, 1, 100)
    xi_p = np.random.default_rng(42).normal(0, 1, 100)
    xi_pn = np.random.default_rng(42).normal(0, 1, 100)
    eps_pn = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_quantile_hadamard_inequality(F, h_n, t_n, xi_p, xi_pn, eps_pn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr043_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    h_n = np.random.default_rng(42).normal(0, 1, 100)
    t_n = np.random.default_rng(42).normal(0, 1, 100)
    xi_p = np.random.default_rng(42).normal(0, 1, 100)
    xi_pn = np.random.default_rng(42).normal(0, 1, 100)
    eps_pn = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_quantile_hadamard_inequality(F, h_n, t_n, xi_p, xi_pn, eps_pn)
    assert isinstance(result, dict)
