"""Tests for airl.airl."""
import numpy as np
import pytest
from moirais.fn.airl import airl


def test_airl_basic():
    """Test basic functionality."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = airl(expert_trajs, D, policy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_airl_edge():
    """Test edge cases."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = airl(expert_trajs, D, policy)
    assert isinstance(result, dict)
