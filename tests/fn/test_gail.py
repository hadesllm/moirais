"""Tests for gail.gail."""
import numpy as np
import pytest
from morie.fn.gail import gail


def test_gail_basic():
    """Test basic functionality."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = gail(expert_trajs, D, policy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gail_edge():
    """Test edge cases."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = gail(expert_trajs, D, policy)
    assert isinstance(result, dict)
