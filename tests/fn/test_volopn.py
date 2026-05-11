"""Tests for volopn.vol_implied_volatility_bs."""
import numpy as np
import pytest
from morie.fn.volopn import vol_implied_volatility_bs


def test_volopn_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    r = 10
    C_obs = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_implied_volatility_bs(S, K, T, r, C_obs, kind)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volopn_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    r = 10
    C_obs = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_implied_volatility_bs(S, K, T, r, C_obs, kind)
    assert isinstance(result, dict)
