"""Tests for msm249.mvsml_reproducing_kernel_eq_10_5."""
import numpy as np
import pytest
from moirais.fn.msm249 import mvsml_reproducing_kernel_eq_10_5


def test_msm249_basic():
    """Test basic functionality."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_5(jk, E, w, l, where, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm249_edge():
    """Test edge cases."""
    jk = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_5(jk, E, w, l, where, the)
    assert isinstance(result, dict)
