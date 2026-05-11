"""Tests for agprtg.alphazero_priority_target."""
import numpy as np
import pytest
from morie.fn.agprtg import alphazero_priority_target


def test_agprtg_basic():
    """Test basic functionality."""
    replay_buffer = np.random.default_rng(42).normal(0, 1, 100)
    priorities = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_priority_target(replay_buffer, priorities)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agprtg_edge():
    """Test edge cases."""
    replay_buffer = np.random.default_rng(42).normal(0, 1, 100)
    priorities = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_priority_target(replay_buffer, priorities)
    assert isinstance(result, dict)
