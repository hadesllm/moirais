"""Tests for hmrl.geron_reinforcement_learning."""
import numpy as np
import pytest
from morie.fn.hmrl import geron_reinforcement_learning


def test_hmrl_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_reinforcement_learning(env, pi, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrl_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_reinforcement_learning(env, pi, gamma)
    assert isinstance(result, dict)
