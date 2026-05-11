"""Tests for evrlci.evt_return_level_ci."""
import numpy as np
import pytest
from morie.fn.evrlci import evt_return_level_ci


def test_evrlci_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    Sigma_hat = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level_ci(mu, sigma, xi, Sigma_hat, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evrlci_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    Sigma_hat = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level_ci(mu, sigma, xi, Sigma_hat, T)
    assert isinstance(result, dict)
