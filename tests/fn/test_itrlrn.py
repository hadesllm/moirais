"""Tests for itrlrn.iterative_q_learning."""
import numpy as np
import pytest
from morie.fn.itrlrn import iterative_q_learning


def test_itrlrn_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = iterative_q_learning(state, action, reward, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itrlrn_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = iterative_q_learning(state, action, reward, time)
    assert isinstance(result, dict)
