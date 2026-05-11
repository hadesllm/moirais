"""Tests for sa_opt.simulated_annealing."""
import numpy as np
import pytest
from morie.fn.sa_opt import simulated_annealing


def test_sa_opt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    T_init = 0.0
    cooling = np.random.default_rng(42).normal(0, 1, 100)
    result = simulated_annealing(f, x0, T_init, cooling)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sa_opt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    T_init = 0.0
    cooling = np.random.default_rng(42).normal(0, 1, 100)
    result = simulated_annealing(f, x0, T_init, cooling)
    assert isinstance(result, dict)
