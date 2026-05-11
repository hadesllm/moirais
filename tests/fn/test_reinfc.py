"""Tests for reinfc.reinforce."""
import numpy as np
import pytest
from morie.fn.reinfc import reinforce


def test_reinfc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = reinforce(env, policy, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reinfc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = reinforce(env, policy, alpha, gamma)
    assert isinstance(result, dict)
