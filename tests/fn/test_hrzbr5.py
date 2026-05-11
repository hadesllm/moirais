"""Tests for hrzbr5.horowitz_bias_reduction_deconv."""
import numpy as np
import pytest
from morie.fn.hrzbr5 import horowitz_bias_reduction_deconv


def test_hrzbr5_basic():
    """Test basic functionality."""
    bandwidth = 0.3
    kernel_order = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_bias_reduction_deconv(bandwidth, kernel_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzbr5_edge():
    """Test edge cases."""
    bandwidth = 0.3
    kernel_order = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_bias_reduction_deconv(bandwidth, kernel_order)
    assert isinstance(result, dict)
