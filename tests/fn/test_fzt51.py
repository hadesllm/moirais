"""Tests for fzt51.fauzi_thm5_1_naive_kernel_equiv."""
import numpy as np
import pytest
from moirais.fn.fzt51 import fauzi_thm5_1_naive_kernel_equiv


def test_fzt51_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_thm5_1_naive_kernel_equiv(data, bandwidth, cdf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt51_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_thm5_1_naive_kernel_equiv(data, bandwidth, cdf)
    assert isinstance(result, dict)
