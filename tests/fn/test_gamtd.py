"""Tests for gamtd.n_step_td."""
import numpy as np
import pytest
from morie.fn.gamtd import n_step_td


def test_gamtd_basic():
    """Test basic functionality."""
    traj = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    gamma = 1.0
    result = n_step_td(traj, V, n, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gamtd_edge():
    """Test edge cases."""
    traj = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    gamma = 1.0
    result = n_step_td(traj, V, n, gamma)
    assert isinstance(result, dict)
