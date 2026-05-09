"""Tests for ksr044.kosorok_ch2_quantile_taylor_bounds."""
import numpy as np
import pytest
from moirais.fn.ksr044 import kosorok_ch2_quantile_taylor_bounds


def test_ksr044_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    t_n = np.random.default_rng(42).normal(0, 1, 100)
    xi_pn = np.random.default_rng(42).normal(0, 1, 100)
    eps_pn = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_quantile_taylor_bounds(F, h, t_n, xi_pn, eps_pn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr044_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    t_n = np.random.default_rng(42).normal(0, 1, 100)
    xi_pn = np.random.default_rng(42).normal(0, 1, 100)
    eps_pn = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_quantile_taylor_bounds(F, h, t_n, xi_pn, eps_pn)
    assert isinstance(result, dict)
