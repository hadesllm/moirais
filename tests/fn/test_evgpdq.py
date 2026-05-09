"""Tests for evgpdq.evt_gpd_quantile."""
import numpy as np
import pytest
from moirais.fn.evgpdq import evt_gpd_quantile


def test_evgpdq_basic():
    """Test basic functionality."""
    p = 5
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_quantile(p, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdq_edge():
    """Test edge cases."""
    p = 5
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_quantile(p, sigma, xi)
    assert isinstance(result, dict)
