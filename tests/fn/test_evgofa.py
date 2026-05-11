"""Tests for evgofa.evt_gev_anderson_darling."""
import numpy as np
import pytest
from morie.fn.evgofa import evt_gev_anderson_darling


def test_evgofa_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_anderson_darling(x, mu, sigma, xi)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_evgofa_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_anderson_darling(x, mu, sigma, xi)
    assert isinstance(result, dict)
