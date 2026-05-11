"""Tests for hmmdp.geron_mdp."""
import numpy as np
import pytest
from morie.fn.hmmdp import geron_mdp


def test_hmmdp_basic():
    """Test basic functionality."""
    states = np.random.default_rng(42).normal(0, 1, 100)
    actions = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_mdp(states, actions, P, R, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmdp_edge():
    """Test edge cases."""
    states = np.random.default_rng(42).normal(0, 1, 100)
    actions = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_mdp(states, actions, P, R, gamma)
    assert isinstance(result, dict)
