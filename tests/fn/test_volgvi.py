"""Tests for volgvi.vol_garch_var_impl."""
import numpy as np
import pytest
from moirais.fn.volgvi import vol_garch_var_impl


def test_volgvi_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    dist = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_var_impl(mu, sigma_next, alpha, dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgvi_edge():
    """Test edge cases."""
    mu = 0.0
    sigma_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    dist = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_var_impl(mu, sigma_next, alpha, dist)
    assert isinstance(result, dict)
