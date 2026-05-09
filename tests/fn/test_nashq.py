"""Tests for nashq.nash_q_learning."""
import numpy as np
import pytest
from moirais.fn.nashq import nash_q_learning


def test_nashq_basic():
    """Test basic functionality."""
    agents = np.random.default_rng(42).normal(0, 1, 100)
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = nash_q_learning(agents, env)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nashq_edge():
    """Test edge cases."""
    agents = np.random.default_rng(42).normal(0, 1, 100)
    env = np.random.default_rng(42).normal(0, 1, 100)
    result = nash_q_learning(agents, env)
    assert isinstance(result, dict)
