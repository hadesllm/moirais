"""Tests for ppoc.ppo."""
import numpy as np
import pytest
from moirais.fn.ppoc import ppo


def test_ppoc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    clip_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = ppo(env, policy, clip_eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ppoc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    clip_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = ppo(env, policy, clip_eps)
    assert isinstance(result, dict)
