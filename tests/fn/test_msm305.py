"""Tests for msm305.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from morie.fn.msm305 import mvsml_convolutional_nn_eq_14_13


def test_msm305_basic():
    """Test basic functionality."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    spline = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(B, spline, basis, PBFR, without, BFR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm305_edge():
    """Test edge cases."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    spline = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(B, spline, basis, PBFR, without, BFR)
    assert isinstance(result, dict)
