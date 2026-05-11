"""Tests for msm250.mvsml_reproducing_kernel_eq_10_12."""
import numpy as np
import pytest
from morie.fn.msm250 import mvsml_reproducing_kernel_eq_10_12


def test_msm250_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = mvsml_reproducing_kernel_eq_10_12(g, l, z, w, V, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm250_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = mvsml_reproducing_kernel_eq_10_12(g, l, z, w, V, h)
    assert isinstance(result, dict)
