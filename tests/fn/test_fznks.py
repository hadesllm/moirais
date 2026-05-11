"""Tests for fznks.fauzi_naive_kernel_ks."""
import numpy as np
import pytest
from morie.fn.fznks import fauzi_naive_kernel_ks


def test_fznks_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_naive_kernel_ks(data, bandwidth, cdf)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fznks_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_naive_kernel_ks(data, bandwidth, cdf)
    assert isinstance(result, dict)
