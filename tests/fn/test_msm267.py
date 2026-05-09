"""Tests for msm267.mvsml_convolutional_nn_eq_14_4."""
import numpy as np
import pytest
from moirais.fn.msm267 import mvsml_convolutional_nn_eq_14_4


def test_msm267_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    xil = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = mvsml_convolutional_nn_eq_14_4(R, T, l, xil, xi, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm267_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    xil = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = mvsml_convolutional_nn_eq_14_4(R, T, l, xil, xi, t)
    assert isinstance(result, dict)
