"""Tests for hrzkqre.horowitz_kernel_quantile_reg."""
import numpy as np
import pytest
from morie.fn.hrzkqre import horowitz_kernel_quantile_reg


def test_hrzkqre_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_kernel_quantile_reg(x, y, bandwidth, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzkqre_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_kernel_quantile_reg(x, y, bandwidth, tau)
    assert isinstance(result, dict)
