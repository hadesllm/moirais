"""Tests for btsbb.boot_stationary_block."""
import numpy as np
import pytest
from moirais.fn.btsbb import boot_stationary_block


def test_btsbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_stationary_block(x, p, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btsbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_stationary_block(x, p, stat, B)
    assert isinstance(result, dict)
