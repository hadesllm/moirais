"""Tests for raindq.rainbow_dqn."""
import numpy as np
import pytest
from moirais.fn.raindq import rainbow_dqn


def test_raindq_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = rainbow_dqn(env)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_raindq_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = rainbow_dqn(env)
    assert isinstance(result, dict)
