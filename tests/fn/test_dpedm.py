"""Tests for dpedm.approx_dp."""
import numpy as np
import pytest
from moirais.fn.dpedm import approx_dp


def test_dpedm_basic():
    """Test basic functionality."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    D_prime = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = approx_dp(mech, D, D_prime, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpedm_edge():
    """Test edge cases."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    D_prime = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = approx_dp(mech, D, D_prime, delta)
    assert isinstance(result, dict)
