"""Tests for evtsthr.evt_threshold_select_lvar."""
import numpy as np
import pytest
from moirais.fn.evtsthr import evt_threshold_select_lvar


def test_evtsthr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_threshold_select_lvar(x, u_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evtsthr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_threshold_select_lvar(x, u_grid)
    assert isinstance(result, dict)
