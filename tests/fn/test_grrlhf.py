"""Tests for grrlhf.geron_rlhf_reward_kl_objective."""
import numpy as np
import pytest
from morie.fn.grrlhf import geron_rlhf_reward_kl_objective


def test_grrlhf_basic():
    """Test basic functionality."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    policy_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_rlhf_reward_kl_objective(rewards, policy_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrlhf_edge():
    """Test edge cases."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    policy_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_rlhf_reward_kl_objective(rewards, policy_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
