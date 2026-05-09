"""Tests for evvarpot.evt_pot_var."""
import numpy as np
import pytest
from moirais.fn.evvarpot import evt_pot_var


def test_evvarpot_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    zeta_u = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = evt_pot_var(u, sigma, xi, zeta_u, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evvarpot_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    zeta_u = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = evt_pot_var(u, sigma, xi, zeta_u, p)
    assert isinstance(result, dict)
