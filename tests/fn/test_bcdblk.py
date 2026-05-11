"""Tests for bcdblk.block_coordinate_descent."""
import numpy as np
import pytest
from morie.fn.bcdblk import block_coordinate_descent


def test_bcdblk_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = block_coordinate_descent(f, blocks, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bcdblk_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = block_coordinate_descent(f, blocks, x0)
    assert isinstance(result, dict)
