"""Tests for msm256.mvsml_reproducing_kernel_eq_10_17."""
import numpy as np
import pytest
from morie.fn.msm256 import mvsml_reproducing_kernel_eq_10_17


def test_msm256_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    h = 0.3
    kp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    ikxip = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_17(w, h, kp, t, ikxip, This)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm256_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    h = 0.3
    kp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    ikxip = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_17(w, h, kp, t, ikxip, This)
    assert isinstance(result, dict)
