"""Tests for hma3c.geron_a3c."""
import numpy as np
import pytest
from moirais.fn.hma3c import geron_a3c


def test_hma3c_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    n_workers = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_a3c(env, actor, critic, n_workers, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hma3c_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic = np.random.default_rng(42).normal(0, 1, 100)
    n_workers = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_a3c(env, actor, critic, n_workers, lr)
    assert isinstance(result, dict)
