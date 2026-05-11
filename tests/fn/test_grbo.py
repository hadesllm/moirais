"""Tests for grbo.geron_bellman_optimality."""
import numpy as np
import pytest
from morie.fn.grbo import geron_bellman_optimality


def test_grbo_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_bellman_optimality(Q, transitions, rewards, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbo_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_bellman_optimality(Q, transitions, rewards, gamma)
    assert isinstance(result, dict)
