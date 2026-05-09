"""Tests for agtmpd.alphazero_temp_decay."""
import numpy as np
import pytest
from moirais.fn.agtmpd import alphazero_temp_decay


def test_agtmpd_basic():
    """Test basic functionality."""
    move_count = 100
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_temp_decay(move_count, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agtmpd_edge():
    """Test edge cases."""
    move_count = 100
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_temp_decay(move_count, threshold)
    assert isinstance(result, dict)
