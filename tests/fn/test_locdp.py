"""Tests for locdp.local_dp."""
import numpy as np
import pytest
from moirais.fn.locdp import local_dp


def test_locdp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mech = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = local_dp(x, mech, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_locdp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mech = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = local_dp(x, mech, epsilon)
    assert isinstance(result, dict)
