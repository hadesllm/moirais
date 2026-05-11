"""Tests for evchibd.evt_chibar_dependence."""
import numpy as np
import pytest
from morie.fn.evchibd import evt_chibar_dependence


def test_evchibd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_chibar_dependence(x, y, u_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evchibd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_chibar_dependence(x, y, u_grid)
    assert isinstance(result, dict)
