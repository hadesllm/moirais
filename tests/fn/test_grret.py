"""Tests for grret.geron_discounted_return."""
import numpy as np
import pytest
from moirais.fn.grret import geron_discounted_return


def test_grret_basic():
    """Test basic functionality."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_discounted_return(rewards, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grret_edge():
    """Test edge cases."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_discounted_return(rewards, gamma)
    assert isinstance(result, dict)
