"""Tests for evgevs.evt_gev_sample."""
import numpy as np
import pytest
from morie.fn.evgevs import evt_gev_sample


def test_evgevs_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = evt_gev_sample(mu, sigma, xi, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevs_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = evt_gev_sample(mu, sigma, xi, n)
    assert isinstance(result, dict)
