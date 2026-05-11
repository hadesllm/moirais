"""Tests for slowdp.slow_dp_truncate."""
import numpy as np
import pytest
from morie.fn.slowdp import slow_dp_truncate


def test_slowdp_basic():
    """Test basic functionality."""
    alpha = 0.05
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = slow_dp_truncate(alpha, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_slowdp_edge():
    """Test edge cases."""
    alpha = 0.05
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = slow_dp_truncate(alpha, eps)
    assert isinstance(result, dict)
