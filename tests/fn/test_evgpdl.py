"""Tests for evgpdl.evt_gpd_loglik."""
import numpy as np
import pytest
from morie.fn.evgpdl import evt_gpd_loglik


def test_evgpdl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_loglik(y, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_loglik(y, sigma, xi)
    assert isinstance(result, dict)
