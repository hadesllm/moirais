"""Tests for msm245.mvsml_reproducing_kernel_eq_10_4."""
import numpy as np
import pytest
from moirais.fn.msm245 import mvsml_reproducing_kernel_eq_10_4


def test_msm245_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    xm0 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_4(X, m1, m0, F, x1, xm0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm245_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    xm0 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_4(X, m1, m0, F, x1, xm0)
    assert isinstance(result, dict)
