"""Tests for evrl.evt_return_level."""
import numpy as np
import pytest
from morie.fn.evrl import evt_return_level


def test_evrl_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evrl_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = evt_return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
