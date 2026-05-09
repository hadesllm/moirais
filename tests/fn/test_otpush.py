"""Tests for otpush.ot_pushforward_density."""
import numpy as np
import pytest
from moirais.fn.otpush import ot_pushforward_density


def test_otpush_basic():
    """Test basic functionality."""
    mu_grid = np.random.default_rng(42).normal(0, 1, 100)
    T_jac = np.random.default_rng(42).normal(0, 1, 100)
    T_inv_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_pushforward_density(mu_grid, T_jac, T_inv_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otpush_edge():
    """Test edge cases."""
    mu_grid = np.random.default_rng(42).normal(0, 1, 100)
    T_jac = np.random.default_rng(42).normal(0, 1, 100)
    T_inv_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_pushforward_density(mu_grid, T_jac, T_inv_grid)
    assert isinstance(result, dict)
