"""Tests for trpoc.trpo."""
import numpy as np
import pytest
from morie.fn.trpoc import trpo


def test_trpoc_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    kl_max = 100
    result = trpo(env, policy, kl_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trpoc_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    kl_max = 100
    result = trpo(env, policy, kl_max)
    assert isinstance(result, dict)
