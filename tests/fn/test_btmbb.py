"""Tests for btmbb.boot_moving_block."""
import numpy as np
import pytest
from moirais.fn.btmbb import boot_moving_block


def test_btmbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_len = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_moving_block(x, block_len, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btmbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_len = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_moving_block(x, block_len, stat, B)
    assert isinstance(result, dict)
