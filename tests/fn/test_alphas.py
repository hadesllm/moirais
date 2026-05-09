"""Tests for alphas.alphazero_self_play."""
import numpy as np
import pytest
from moirais.fn.alphas import alphazero_self_play


def test_alphas_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    mcts_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_play(state, policy, value, mcts_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alphas_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    policy = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    mcts_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_play(state, policy, value, mcts_iter)
    assert isinstance(result, dict)
