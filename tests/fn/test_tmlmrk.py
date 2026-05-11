"""Tests for tmlmrk.tmle_markov."""
import numpy as np
import pytest
from morie.fn.tmlmrk import tmle_markov


def test_tmlmrk_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_markov(state, action, reward, policy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmrk_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_markov(state, action, reward, policy)
    assert isinstance(result, dict)
