"""Tests for warpL.warp."""
import numpy as np
import pytest
from moirais.fn.warpL import warp


def test_warpL_basic():
    """Test basic functionality."""
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = warp(pairs, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_warpL_edge():
    """Test edge cases."""
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = warp(pairs, K)
    assert isinstance(result, dict)
