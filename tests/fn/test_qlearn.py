"""Tests for qlearn.q_learning."""
import numpy as np
import pytest
from moirais.fn.qlearn import q_learning


def test_qlearn_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    epsilon = 1e-6
    n_episodes = np.random.default_rng(42).normal(0, 1, 100)
    result = q_learning(env, alpha, gamma, epsilon, n_episodes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qlearn_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    epsilon = 1e-6
    n_episodes = np.random.default_rng(42).normal(0, 1, 100)
    result = q_learning(env, alpha, gamma, epsilon, n_episodes)
    assert isinstance(result, dict)
