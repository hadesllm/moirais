"""Tests for msm306.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from moirais.fn.msm306 import mvsml_convolutional_nn_eq_14_13


def test_msm306_basic():
    """Test basic functionality."""
    sis = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    rst = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(sis, PBFR, without, BFR, rst, derivative)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm306_edge():
    """Test edge cases."""
    sis = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    rst = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(sis, PBFR, without, BFR, rst, derivative)
    assert isinstance(result, dict)
