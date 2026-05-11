"""Tests for evprmstab.evt_param_stability."""
import numpy as np
import pytest
from morie.fn.evprmstab import evt_param_stability


def test_evprmstab_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_param_stability(x, u_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evprmstab_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_param_stability(x, u_grid)
    assert isinstance(result, dict)
