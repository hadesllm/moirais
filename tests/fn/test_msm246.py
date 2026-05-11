"""Tests for msm246.mvsml_reproducing_kernel_eq_10_10."""
import numpy as np
import pytest
from morie.fn.msm246 import mvsml_reproducing_kernel_eq_10_10


def test_msm246_basic():
    """Test basic functionality."""
    nding = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    optimal = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_10(nding, the, optimal, weights, biases, This)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm246_edge():
    """Test edge cases."""
    nding = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    optimal = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_reproducing_kernel_eq_10_10(nding, the, optimal, weights, biases, This)
    assert isinstance(result, dict)
