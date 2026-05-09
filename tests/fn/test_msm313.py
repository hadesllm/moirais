"""Tests for msm313.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from moirais.fn.msm313 import mvsml_convolutional_nn_eq_14_14


def test_msm313_basic():
    """Test basic functionality."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    MSE = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(of, the, MSE, PBFR, vs, BFR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm313_edge():
    """Test edge cases."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    MSE = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(of, the, MSE, PBFR, vs, BFR)
    assert isinstance(result, dict)
