"""Tests for grddqn.geron_double_dqn_target."""
import numpy as np
import pytest
from morie.fn.grddqn import geron_double_dqn_target


def test_grddqn_basic():
    """Test basic functionality."""
    Q_online = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    gamma = 1.0
    result = geron_double_dqn_target(Q_online, Q_target, s_next, r, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grddqn_edge():
    """Test edge cases."""
    Q_online = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    gamma = 1.0
    result = geron_double_dqn_target(Q_online, Q_target, s_next, r, gamma)
    assert isinstance(result, dict)
