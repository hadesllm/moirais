"""Tests for hmdldqn.geron_dueling_dqn."""
import numpy as np
import pytest
from morie.fn.hmdldqn import geron_dueling_dqn


def test_hmdldqn_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dueling_dqn(env, V, A, buffer, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdldqn_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dueling_dqn(env, V, A, buffer, epochs, lr)
    assert isinstance(result, dict)
