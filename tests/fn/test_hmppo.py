"""Tests for hmppo.geron_ppo."""
import numpy as np
import pytest
from moirais.fn.hmppo import geron_ppo


def test_hmppo_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    clip_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ppo(env, policy, epochs, lr, clip_eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmppo_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    clip_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ppo(env, policy, epochs, lr, clip_eps)
    assert isinstance(result, dict)
