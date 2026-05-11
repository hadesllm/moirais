"""Tests for taulep.tau_leap_sim."""
import numpy as np
import pytest
from morie.fn.taulep import tau_leap_sim


def test_taulep_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    rates = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = tau_leap_sim(state, rates, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_taulep_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    rates = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = tau_leap_sim(state, rates, tau)
    assert isinstance(result, dict)
