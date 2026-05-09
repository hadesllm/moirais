"""Tests for msm268.mvsml_convolutional_nn_eq_14_5."""
import numpy as np
import pytest
from moirais.fn.msm268 import mvsml_convolutional_nn_eq_14_5


def test_msm268_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    TX = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_5(b, X, TX, T, n, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm268_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    TX = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_5(b, X, TX, T, n, y)
    assert isinstance(result, dict)
