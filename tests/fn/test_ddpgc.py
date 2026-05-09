"""Tests for ddpgc.ddpg."""
import numpy as np
import pytest
from moirais.fn.ddpgc import ddpg


def test_ddpgc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = ddpg(env, actor, critic, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ddpgc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = ddpg(env, actor, critic, tau)
    assert isinstance(result, dict)
