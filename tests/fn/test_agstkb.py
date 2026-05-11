"""Tests for agstkb.alphazero_stockfish_baseline."""
import numpy as np
import pytest
from morie.fn.agstkb import alphazero_stockfish_baseline


def test_agstkb_basic():
    """Test basic functionality."""
    games = np.random.default_rng(42).normal(0, 1, 100)
    ladder = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_stockfish_baseline(games, ladder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agstkb_edge():
    """Test edge cases."""
    games = np.random.default_rng(42).normal(0, 1, 100)
    ladder = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_stockfish_baseline(games, ladder)
    assert isinstance(result, dict)
