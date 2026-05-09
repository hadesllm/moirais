"""Tests for sarsa.sarsa."""
import numpy as np
import pytest
from moirais.fn.sarsa import sarsa


def test_sarsa_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    epsilon = 1e-6
    n_episodes = np.random.default_rng(42).normal(0, 1, 100)
    result = sarsa(env, alpha, gamma, epsilon, n_episodes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sarsa_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    epsilon = 1e-6
    n_episodes = np.random.default_rng(42).normal(0, 1, 100)
    result = sarsa(env, alpha, gamma, epsilon, n_episodes)
    assert isinstance(result, dict)
