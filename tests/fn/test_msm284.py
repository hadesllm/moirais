"""Tests for msm284.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from moirais.fn.msm284 import mvsml_convolutional_nn_eq_14_12


def test_msm284_basic():
    """Test basic functionality."""
    SSE = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(SSE, j, where, X, T, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm284_edge():
    """Test edge cases."""
    SSE = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(SSE, j, where, X, T, P)
    assert isinstance(result, dict)
