"""Tests for evgofg.evt_gpd_anderson_darling."""
import numpy as np
import pytest
from moirais.fn.evgofg import evt_gpd_anderson_darling


def test_evgofg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_anderson_darling(y, sigma, xi)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_evgofg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_anderson_darling(y, sigma, xi)
    assert isinstance(result, dict)
