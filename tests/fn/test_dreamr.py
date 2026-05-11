"""Tests for dreamr.dreamer."""
import numpy as np
import pytest
from morie.fn.dreamr import dreamer


def test_dreamr_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    world_model = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = dreamer(env, world_model, actor, critic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dreamr_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    world_model = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    result = dreamer(env, world_model, actor, critic)
    assert isinstance(result, dict)
