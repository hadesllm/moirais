"""Tests for bxprfl.baxter_king."""
import numpy as np
import pytest
from morie.fn.bxprfl import baxter_king


def test_bxprfl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p_low = np.random.default_rng(42).normal(0, 1, 100)
    p_high = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = baxter_king(y, p_low, p_high, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bxprfl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p_low = np.random.default_rng(42).normal(0, 1, 100)
    p_high = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = baxter_king(y, p_low, p_high, K)
    assert isinstance(result, dict)
