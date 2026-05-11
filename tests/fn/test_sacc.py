"""Tests for sacc.sac."""
import numpy as np
import pytest
from morie.fn.sacc import sac


def test_sacc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = sac(env, actor, critic, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sacc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = sac(env, actor, critic, alpha)
    assert isinstance(result, dict)
