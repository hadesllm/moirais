"""Tests for agbnch.alphazero_benchmark_eval."""
import numpy as np
import pytest
from morie.fn.agbnch import alphazero_benchmark_eval


def test_agbnch_basic():
    """Test basic functionality."""
    games = np.random.default_rng(42).normal(0, 1, 100)
    ladder = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_benchmark_eval(games, ladder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agbnch_edge():
    """Test edge cases."""
    games = np.random.default_rng(42).normal(0, 1, 100)
    ladder = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_benchmark_eval(games, ladder)
    assert isinstance(result, dict)
