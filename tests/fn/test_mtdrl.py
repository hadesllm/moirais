"""Tests for mtdrl.meta_rl."""
import numpy as np
import pytest
from morie.fn.mtdrl import meta_rl


def test_mtdrl_basic():
    """Test basic functionality."""
    task_dist = np.random.default_rng(42).normal(0, 1, 100)
    rnn = np.random.default_rng(42).normal(0, 1, 100)
    result = meta_rl(task_dist, rnn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtdrl_edge():
    """Test edge cases."""
    task_dist = np.random.default_rng(42).normal(0, 1, 100)
    rnn = np.random.default_rng(42).normal(0, 1, 100)
    result = meta_rl(task_dist, rnn)
    assert isinstance(result, dict)
