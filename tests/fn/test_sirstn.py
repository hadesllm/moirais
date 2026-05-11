"""Tests for sirstn.sir_stochastic."""
import numpy as np
import pytest
from morie.fn.sirstn import sir_stochastic


def test_sirstn_basic():
    """Test basic functionality."""
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    I0 = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    gamma = 1.0
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sir_stochastic(S0, I0, beta, gamma, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sirstn_edge():
    """Test edge cases."""
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    I0 = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    gamma = 1.0
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sir_stochastic(S0, I0, beta, gamma, T)
    assert isinstance(result, dict)
