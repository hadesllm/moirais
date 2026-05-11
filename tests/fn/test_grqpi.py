"""Tests for grqpi.geron_action_value_function."""
import numpy as np
import pytest
from morie.fn.grqpi import geron_action_value_function


def test_grqpi_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_action_value_function(state, action, policy, transitions, rewards, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grqpi_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_action_value_function(state, action, policy, transitions, rewards, gamma)
    assert isinstance(result, dict)
