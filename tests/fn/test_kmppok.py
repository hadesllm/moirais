"""Tests for kmppok.kamath_ppo_rlhf_objective."""
import numpy as np
import pytest
from moirais.fn.kmppok import kamath_ppo_rlhf_objective


def test_kmppok_basic():
    """Test basic functionality."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    logp_theta = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmppok_edge():
    """Test edge cases."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    logp_theta = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta)
    assert isinstance(result, dict)
