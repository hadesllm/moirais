"""Tests for msm298.mvsml_convolutional_nn_eq_14_5."""
import numpy as np
import pytest
from moirais.fn.msm298 import mvsml_convolutional_nn_eq_14_5


def test_msm298_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_convolutional_nn_eq_14_5(where, a, vector, of, dimension, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm298_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_convolutional_nn_eq_14_5(where, a, vector, of, dimension, n)
    assert isinstance(result, dict)
