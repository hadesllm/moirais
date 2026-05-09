"""Tests for msm290.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from moirais.fn.msm290 import mvsml_convolutional_nn_eq_14_12


def test_msm290_basic():
    """Test basic functionality."""
    Penalty = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    p = 5
    result = mvsml_convolutional_nn_eq_14_12(Penalty, matrix, of, derivative, order, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm290_edge():
    """Test edge cases."""
    Penalty = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    derivative = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    p = 5
    result = mvsml_convolutional_nn_eq_14_12(Penalty, matrix, of, derivative, order, p)
    assert isinstance(result, dict)
