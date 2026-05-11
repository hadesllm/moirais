"""Tests for kmrsft.kamath_rejection_sampling_finetune."""
import numpy as np
import pytest
from morie.fn.kmrsft import kamath_rejection_sampling_finetune


def test_kmrsft_basic():
    """Test basic functionality."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    samples = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_rejection_sampling_finetune(prompts, samples, rewards, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrsft_edge():
    """Test edge cases."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    samples = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_rejection_sampling_finetune(prompts, samples, rewards, k)
    assert isinstance(result, dict)
