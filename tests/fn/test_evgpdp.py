"""Tests for evgpdp.evt_gpd_pdf."""
import numpy as np
import pytest
from moirais.fn.evgpdp import evt_gpd_pdf


def test_evgpdp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_pdf(y, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_pdf(y, sigma, xi)
    assert isinstance(result, dict)
