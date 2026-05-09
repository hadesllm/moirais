"""Tests for agselp.alphazero_self_play_eval."""
import numpy as np
import pytest
from moirais.fn.agselp import alphazero_self_play_eval


def test_agselp_basic():
    """Test basic functionality."""
    new_net = np.random.default_rng(42).normal(0, 1, 100)
    old_net = np.random.default_rng(42).normal(0, 1, 100)
    n_games = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_play_eval(new_net, old_net, n_games)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agselp_edge():
    """Test edge cases."""
    new_net = np.random.default_rng(42).normal(0, 1, 100)
    old_net = np.random.default_rng(42).normal(0, 1, 100)
    n_games = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_self_play_eval(new_net, old_net, n_games)
    assert isinstance(result, dict)
