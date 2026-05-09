"""Tests for rdksrn.sharp_rdd."""
import numpy as np
import pytest
from moirais.fn.rdksrn import sharp_rdd


def test_rdksrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bandwidth = 0.3
    result = sharp_rdd(y, x, cutoff, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdksrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bandwidth = 0.3
    result = sharp_rdd(y, x, cutoff, bandwidth)
    assert isinstance(result, dict)
