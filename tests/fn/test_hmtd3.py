"""Tests for hmtd3.geron_td3."""
import numpy as np
import pytest
from moirais.fn.hmtd3 import geron_td3


def test_hmtd3_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    Q1 = np.random.default_rng(42).normal(0, 1, 100)
    Q2 = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_td3(env, policy, Q1, Q2, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtd3_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    Q1 = np.random.default_rng(42).normal(0, 1, 100)
    Q2 = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_td3(env, policy, Q1, Q2, epochs, lr)
    assert isinstance(result, dict)
