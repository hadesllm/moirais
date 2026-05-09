"""Tests for hmrlhf.geron_rlhf."""
import numpy as np
import pytest
from moirais.fn.hmrlhf import geron_rlhf


def test_hmrlhf_basic():
    """Test basic functionality."""
    policy = np.random.default_rng(42).normal(0, 1, 100)
    reward_model = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rlhf(policy, reward_model, prompts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrlhf_edge():
    """Test edge cases."""
    policy = np.random.default_rng(42).normal(0, 1, 100)
    reward_model = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rlhf(policy, reward_model, prompts)
    assert isinstance(result, dict)
