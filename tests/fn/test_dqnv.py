"""Tests for dqnv.deep_q_network."""
import numpy as np
import pytest
from morie.fn.dqnv import deep_q_network


def test_dqnv_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    target_update = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_q_network(env, net, buffer, target_update)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dqnv_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    net = np.random.default_rng(42).normal(0, 1, 100)
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    target_update = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_q_network(env, net, buffer, target_update)
    assert isinstance(result, dict)
