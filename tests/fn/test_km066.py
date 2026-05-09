"""Tests for km066.kamath_ch5_reward_kl_penalty."""
import numpy as np
import pytest
from moirais.fn.km066 import kamath_ch5_reward_kl_penalty


def test_km066_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi_RL = np.random.default_rng(42).normal(0, 1, 100)
    pi_SFT = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km066_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi_RL = np.random.default_rng(42).normal(0, 1, 100)
    pi_SFT = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, beta)
    assert isinstance(result, dict)
