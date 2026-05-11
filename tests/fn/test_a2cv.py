"""Tests for a2cv.a2c."""
import numpy as np
import pytest
from morie.fn.a2cv import a2c


def test_a2cv_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    n_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = a2c(env, actor, critic, n_steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_a2cv_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    n_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = a2c(env, actor, critic, n_steps)
    assert isinstance(result, dict)
