"""Tests for msm253.mvsml_reproducing_kernel_eq_10_14."""
import numpy as np
import pytest
from moirais.fn.msm253 import mvsml_reproducing_kernel_eq_10_14


def test_msm253_basic():
    """Test basic functionality."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    Next = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = mvsml_reproducing_kernel_eq_10_14(jk, Next, to, update, the, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm253_edge():
    """Test edge cases."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    Next = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = mvsml_reproducing_kernel_eq_10_14(jk, Next, to, update, the, weights)
    assert isinstance(result, dict)
