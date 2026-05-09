"""Tests for spgwrk.schabenberger_gwr_kernels."""
import numpy as np
import pytest
from moirais.fn.spgwrk import schabenberger_gwr_kernels


def test_spgwrk_basic():
    """Test basic functionality."""
    distance = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel_type = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_gwr_kernels(distance, bandwidth, kernel_type)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgwrk_edge():
    """Test edge cases."""
    distance = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel_type = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_gwr_kernels(distance, bandwidth, kernel_type)
    assert isinstance(result, dict)
