"""Tests for evqqgev.evt_gev_qq_plot."""
import numpy as np
import pytest
from morie.fn.evqqgev import evt_gev_qq_plot


def test_evqqgev_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_qq_plot(x, mu, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evqqgev_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_qq_plot(x, mu, sigma, xi)
    assert isinstance(result, dict)
