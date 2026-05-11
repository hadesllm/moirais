"""Tests for grppo.geron_ppo_clipped_objective."""
import numpy as np
import pytest
from morie.fn.grppo import geron_ppo_clipped_objective


def test_grppo_basic():
    """Test basic functionality."""
    ratios = np.random.default_rng(42).normal(0, 1, 100)
    advantages = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ppo_clipped_objective(ratios, advantages, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grppo_edge():
    """Test edge cases."""
    ratios = np.random.default_rng(42).normal(0, 1, 100)
    advantages = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ppo_clipped_objective(ratios, advantages, eps)
    assert isinstance(result, dict)
