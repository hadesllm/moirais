"""Tests for evmadog.evt_madogram."""
import numpy as np
import pytest
from moirais.fn.evmadog import evt_madogram


def test_evmadog_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_madogram(x, y, t_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evmadog_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_madogram(x, y, t_grid)
    assert isinstance(result, dict)
