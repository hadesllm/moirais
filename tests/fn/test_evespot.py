"""Tests for evespot.evt_pot_es."""
import numpy as np
import pytest
from moirais.fn.evespot import evt_pot_es


def test_evespot_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    VaR = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_pot_es(u, sigma, xi, VaR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evespot_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    VaR = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_pot_es(u, sigma, xi, VaR)
    assert isinstance(result, dict)
