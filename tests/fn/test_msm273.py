"""Tests for msm273.mvsml_convolutional_nn_eq_14_4."""
import numpy as np
import pytest
from morie.fn.msm273 import mvsml_convolutional_nn_eq_14_4


def test_msm273_basic():
    """Test basic functionality."""
    n = 100
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    Finally = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_4(n, X, T, x1, t, Finally)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm273_edge():
    """Test edge cases."""
    n = 100
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    Finally = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_4(n, X, T, x1, t, Finally)
    assert isinstance(result, dict)
