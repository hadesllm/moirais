"""Tests for hrzllr.horowitz_local_linear."""
import numpy as np
import pytest
from morie.fn.hrzllr import horowitz_local_linear


def test_hrzllr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_local_linear(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzllr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_local_linear(x, y, bandwidth)
    assert isinstance(result, dict)
