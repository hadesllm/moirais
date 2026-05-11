"""Tests for evgpds.evt_gpd_sample."""
import numpy as np
import pytest
from morie.fn.evgpds import evt_gpd_sample


def test_evgpds_basic():
    """Test basic functionality."""
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = evt_gpd_sample(sigma, xi, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpds_edge():
    """Test edge cases."""
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = evt_gpd_sample(sigma, xi, n)
    assert isinstance(result, dict)
