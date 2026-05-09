"""Tests for safrl.safe_rl."""
import numpy as np
import pytest
from moirais.fn.safrl import safe_rl


def test_safrl_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    cost_fn = (lambda v: v)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = safe_rl(env, policy, cost_fn, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_safrl_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    cost_fn = (lambda v: v)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = safe_rl(env, policy, cost_fn, threshold)
    assert isinstance(result, dict)
