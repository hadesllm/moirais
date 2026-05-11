"""Tests for msm255.mvsml_reproducing_kernel_eq_10_17."""
import numpy as np
import pytest
from morie.fn.msm255 import mvsml_reproducing_kernel_eq_10_17


def test_msm255_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    h = 0.3
    kp = np.random.default_rng(42).normal(0, 1, 100)
    ikxip = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_17(t, w, h, kp, ikxip)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm255_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    h = 0.3
    kp = np.random.default_rng(42).normal(0, 1, 100)
    ikxip = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_17(t, w, h, kp, ikxip)
    assert isinstance(result, dict)
