"""Tests for mctsel.mcts_selection."""
import numpy as np
import pytest
from morie.fn.mctsel import mcts_selection


def test_mctsel_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    P = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = mcts_selection(Q, N, P, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mctsel_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    P = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = mcts_selection(Q, N, P, c)
    assert isinstance(result, dict)
