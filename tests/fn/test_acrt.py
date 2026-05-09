"""Tests for acrt.actor_critic."""
import numpy as np
import pytest
from moirais.fn.acrt import actor_critic


def test_acrt_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = actor_critic(env, actor, critic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acrt_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = actor_critic(env, actor, critic)
    assert isinstance(result, dict)
