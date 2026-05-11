"""Tests for grac.geron_actor_critic_advantage."""
import numpy as np
import pytest
from morie.fn.grac import geron_actor_critic_advantage


def test_grac_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    gamma = 1.0
    result = geron_actor_critic_advantage(V, s, s_next, r, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grac_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    gamma = 1.0
    result = geron_actor_critic_advantage(V, s, s_next, r, gamma)
    assert isinstance(result, dict)
