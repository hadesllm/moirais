"""Tests for hmrnfc.geron_reinforce."""
import numpy as np
import pytest
from moirais.fn.hmrnfc import geron_reinforce


def test_hmrnfc_basic():
    """Test basic functionality."""
    episodes = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reinforce(episodes, policy, gamma, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrnfc_edge():
    """Test edge cases."""
    episodes = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reinforce(episodes, policy, gamma, eta)
    assert isinstance(result, dict)
