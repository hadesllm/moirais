"""Tests for rdkkin.kink_rdd."""
import numpy as np
import pytest
from moirais.fn.rdkkin import kink_rdd


def test_rdkkin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bandwidth = 0.3
    result = kink_rdd(y, x, cutoff, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdkkin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bandwidth = 0.3
    result = kink_rdd(y, x, cutoff, bandwidth)
    assert isinstance(result, dict)
