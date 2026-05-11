"""Tests for evgpdc.evt_gpd_cdf."""
import numpy as np
import pytest
from morie.fn.evgpdc import evt_gpd_cdf


def test_evgpdc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_cdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_cdf(y, sigma, xi)
    assert isinstance(result, dict)
