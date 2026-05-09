"""Tests for hmddqn.geron_double_dqn."""
import numpy as np
import pytest
from moirais.fn.hmddqn import geron_double_dqn


def test_hmddqn_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_double_dqn(env, Q, Q_target, buffer, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmddqn_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_double_dqn(env, Q, Q_target, buffer, epochs, lr)
    assert isinstance(result, dict)
