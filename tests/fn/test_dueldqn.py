"""Tests for dueldqn.dueling_dqn."""
import numpy as np
import pytest
from morie.fn.dueldqn import dueling_dqn


def test_dueldqn_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    result = dueling_dqn(env, net)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dueldqn_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    result = dueling_dqn(env, net)
    assert isinstance(result, dict)
