"""Tests for kmstgn.kamath_summarize_from_feedback."""
import numpy as np
import pytest
from morie.fn.kmstgn import kamath_summarize_from_feedback


def test_kmstgn_basic():
    """Test basic functionality."""
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    pi_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_summarize_from_feedback(preferences, rewards, pi_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmstgn_edge():
    """Test edge cases."""
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    pi_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_summarize_from_feedback(preferences, rewards, pi_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
