"""Tests for rlhfRS.rlhf_recommendation."""
import numpy as np
import pytest
from moirais.fn.rlhfRS import rlhf_recommendation


def test_rlhfRS_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = rlhf_recommendation(env, policy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rlhfRS_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = rlhf_recommendation(env, policy)
    assert isinstance(result, dict)
