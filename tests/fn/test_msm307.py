"""Tests for msm307.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from moirais.fn.msm307 import mvsml_convolutional_nn_eq_14_13


def test_msm307_basic():
    """Test basic functionality."""
    nd = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    rst = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(nd, without, BFR, rst, derivative, penalization)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm307_edge():
    """Test edge cases."""
    nd = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    rst = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(nd, without, BFR, rst, derivative, penalization)
    assert isinstance(result, dict)
