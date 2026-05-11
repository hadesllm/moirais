"""Tests for msm280.mvsml_convolutional_nn_eq_14_10."""
import numpy as np
import pytest
from morie.fn.msm280 import mvsml_convolutional_nn_eq_14_10


def test_msm280_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    square = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    entries = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(where, P, a, square, matrix, entries)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm280_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    square = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    entries = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(where, P, a, square, matrix, entries)
    assert isinstance(result, dict)
