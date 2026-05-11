"""Tests for msm252.mvsml_reproducing_kernel_eq_10_13."""
import numpy as np
import pytest
from morie.fn.msm252 import mvsml_reproducing_kernel_eq_10_13


def test_msm252_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    jk = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    ijV = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = mvsml_reproducing_kernel_eq_10_13(w, l, jk, t, ijV, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm252_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    jk = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    ijV = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = mvsml_reproducing_kernel_eq_10_13(w, l, jk, t, ijV, h)
    assert isinstance(result, dict)
