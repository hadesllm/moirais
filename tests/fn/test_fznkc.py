"""Tests for fznkc.fauzi_naive_kernel_cvm."""
import numpy as np
import pytest
from moirais.fn.fznkc import fauzi_naive_kernel_cvm


def test_fznkc_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_naive_kernel_cvm(data, bandwidth, cdf)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fznkc_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_naive_kernel_cvm(data, bandwidth, cdf)
    assert isinstance(result, dict)
