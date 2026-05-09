"""Tests for evgevp.evt_gev_pdf."""
import numpy as np
import pytest
from moirais.fn.evgevp import evt_gev_pdf


def test_evgevp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pdf(x, mu, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pdf(x, mu, sigma, xi)
    assert isinstance(result, dict)
