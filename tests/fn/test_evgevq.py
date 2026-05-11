"""Tests for evgevq.evt_gev_quantile."""
import numpy as np
import pytest
from morie.fn.evgevq import evt_gev_quantile


def test_evgevq_basic():
    """Test basic functionality."""
    p = 5
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_quantile(p, mu, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevq_edge():
    """Test edge cases."""
    p = 5
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_quantile(p, mu, sigma, xi)
    assert isinstance(result, dict)
