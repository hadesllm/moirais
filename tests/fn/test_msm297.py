"""Tests for msm297.mvsml_convolutional_nn_eq_14_4."""
import numpy as np
import pytest
from morie.fn.msm297 import mvsml_convolutional_nn_eq_14_4


def test_msm297_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_convolutional_nn_eq_14_4(where, a, vector, of, dimension, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm297_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_convolutional_nn_eq_14_4(where, a, vector, of, dimension, n)
    assert isinstance(result, dict)
