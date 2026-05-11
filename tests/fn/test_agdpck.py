"""Tests for agdpck.alphazero_data_pickle."""
import numpy as np
import pytest
from morie.fn.agdpck import alphazero_data_pickle


def test_agdpck_basic():
    """Test basic functionality."""
    replay_buffer = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_data_pickle(replay_buffer, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agdpck_edge():
    """Test edge cases."""
    replay_buffer = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_data_pickle(replay_buffer, path)
    assert isinstance(result, dict)
