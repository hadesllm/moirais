"""Tests for hma2c.geron_a2c."""
import numpy as np
import pytest
from moirais.fn.hma2c import geron_a2c


def test_hma2c_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_a2c(env, actor, critic, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hma2c_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_a2c(env, actor, critic, epochs, lr)
    assert isinstance(result, dict)
