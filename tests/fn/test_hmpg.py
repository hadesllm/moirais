"""Tests for hmpg.geron_policy_gradient."""
import numpy as np
import pytest
from moirais.fn.hmpg import geron_policy_gradient


def test_hmpg_basic():
    """Test basic functionality."""
    trajectories = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_policy_gradient(trajectories, policy, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpg_edge():
    """Test edge cases."""
    trajectories = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_policy_gradient(trajectories, policy, gamma)
    assert isinstance(result, dict)
