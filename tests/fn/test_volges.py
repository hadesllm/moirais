"""Tests for volges.vol_garch_es_impl."""
import numpy as np
import pytest
from moirais.fn.volges import vol_garch_es_impl


def test_volges_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    dist = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_es_impl(mu, sigma_next, alpha, dist, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volges_edge():
    """Test edge cases."""
    mu = 0.0
    sigma_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    dist = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_es_impl(mu, sigma_next, alpha, dist, nu)
    assert isinstance(result, dict)
