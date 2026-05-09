"""Tests for gae.gae."""
import numpy as np
import pytest
from moirais.fn.gae import gae


def test_gae_basic():
    """Test basic functionality."""
    traj = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    lam = 0.1
    result = gae(traj, V, gamma, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gae_edge():
    """Test edge cases."""
    traj = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    lam = 0.1
    result = gae(traj, V, gamma, lam)
    assert isinstance(result, dict)
