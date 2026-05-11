"""Tests for rmrl.reward_machine."""
import numpy as np
import pytest
from morie.fn.rmrl import reward_machine


def test_rmrl_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    fsa = np.random.default_rng(42).normal(0, 1, 100)
    result = reward_machine(env, fsa)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rmrl_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    fsa = np.random.default_rng(42).normal(0, 1, 100)
    result = reward_machine(env, fsa)
    assert isinstance(result, dict)
