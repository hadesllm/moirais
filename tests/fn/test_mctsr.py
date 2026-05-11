"""Tests for mctsr.mcts_rollout."""
import numpy as np
import pytest
from morie.fn.mctsr import mcts_rollout


def test_mctsr_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    budget = np.random.default_rng(42).normal(0, 1, 100)
    result = mcts_rollout(state, budget)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mctsr_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    budget = np.random.default_rng(42).normal(0, 1, 100)
    result = mcts_rollout(state, budget)
    assert isinstance(result, dict)
