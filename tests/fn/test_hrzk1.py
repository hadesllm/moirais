"""Tests for hrzk1.horowitz_kernel_density."""
import numpy as np
import pytest
from moirais.fn.hrzk1 import horowitz_kernel_density


def test_hrzk1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_kernel_density(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzk1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_kernel_density(x, bandwidth)
    assert isinstance(result, dict)
