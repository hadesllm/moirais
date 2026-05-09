"""Tests for msm248.mvsml_reproducing_kernel_eq_10_9."""
import numpy as np
import pytest
from moirais.fn.msm248 import mvsml_reproducing_kernel_eq_10_9


def test_msm248_basic():
    """Test basic functionality."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_9(jk, E, w, l, where, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm248_edge():
    """Test edge cases."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_9(jk, E, w, l, where, the)
    assert isinstance(result, dict)
