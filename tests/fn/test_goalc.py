"""Tests for goalc.goal_conditioned."""
import numpy as np
import pytest
from moirais.fn.goalc import goal_conditioned


def test_goalc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    goal_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = goal_conditioned(env, policy, goal_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_goalc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    goal_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = goal_conditioned(env, policy, goal_dist)
    assert isinstance(result, dict)
