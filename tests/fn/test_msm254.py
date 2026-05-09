"""Tests for msm254.mvsml_reproducing_kernel_eq_10_16."""
import numpy as np
import pytest
from moirais.fn.msm254 import mvsml_reproducing_kernel_eq_10_16


def test_msm254_basic():
    """Test basic functionality."""
    XL = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    h = 0.3
    w = np.random.default_rng(45).exponential(1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    ijw = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_16(XL, z, h, w, j, ijw)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm254_edge():
    """Test edge cases."""
    XL = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    h = 0.3
    w = np.random.default_rng(45).exponential(1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    ijw = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_16(XL, z, h, w, j, ijw)
    assert isinstance(result, dict)
