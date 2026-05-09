"""Tests for evrlpot.evt_return_level_pot."""
import numpy as np
import pytest
from moirais.fn.evrlpot import evt_return_level_pot


def test_evrlpot_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    zeta_u = np.random.default_rng(42).normal(0, 1, 100)
    n_y = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level_pot(u, sigma, xi, zeta_u, n_y, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evrlpot_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    zeta_u = np.random.default_rng(42).normal(0, 1, 100)
    n_y = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level_pot(u, sigma, xi, zeta_u, n_y, T)
    assert isinstance(result, dict)
