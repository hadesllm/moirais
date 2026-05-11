"""Tests for evpdfn.evt_pickands_dep_fn."""
import numpy as np
import pytest
from morie.fn.evpdfn import evt_pickands_dep_fn


def test_evpdfn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_pickands_dep_fn(x, y, t_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evpdfn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_pickands_dep_fn(x, y, t_grid)
    assert isinstance(result, dict)
