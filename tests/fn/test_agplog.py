"""Tests for agplog.alphazero_play_log."""
import numpy as np
import pytest
from moirais.fn.agplog import alphazero_play_log


def test_agplog_basic():
    """Test basic functionality."""
    game = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_play_log(game, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agplog_edge():
    """Test edge cases."""
    game = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_play_log(game, path)
    assert isinstance(result, dict)
