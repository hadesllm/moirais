"""Tests for dpepsm.epsilon_dp."""
import numpy as np
import pytest
from moirais.fn.dpepsm import epsilon_dp


def test_dpepsm_basic():
    """Test basic functionality."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    D_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_dp(mech, D, D_prime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpepsm_edge():
    """Test edge cases."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    D_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_dp(mech, D, D_prime)
    assert isinstance(result, dict)
