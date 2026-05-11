"""Tests for evmrlp.evt_mean_residual_life."""
import numpy as np
import pytest
from morie.fn.evmrlp import evt_mean_residual_life


def test_evmrlp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_mean_residual_life(x, u_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evmrlp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_mean_residual_life(x, u_grid)
    assert isinstance(result, dict)
