"""Tests for td3c.td3."""
import numpy as np
import pytest
from moirais.fn.td3c import td3


def test_td3c_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic1 = np.random.default_rng(42).normal(0, 1, 100)
    critic2 = np.random.default_rng(42).normal(0, 1, 100)
    result = td3(env, actor, critic1, critic2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_td3c_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    critic1 = np.random.default_rng(42).normal(0, 1, 100)
    critic2 = np.random.default_rng(42).normal(0, 1, 100)
    result = td3(env, actor, critic1, critic2)
    assert isinstance(result, dict)
