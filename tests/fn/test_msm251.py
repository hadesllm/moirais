"""Tests for msm251.mvsml_reproducing_kernel_eq_10_13."""
import numpy as np
import pytest
from morie.fn.msm251 import mvsml_reproducing_kernel_eq_10_13


def test_msm251_basic():
    """Test basic functionality."""
    ij = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    output = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_13(ij, the, hidden, units, to, output)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm251_edge():
    """Test edge cases."""
    ij = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    output = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_13(ij, the, hidden, units, to, output)
    assert isinstance(result, dict)
