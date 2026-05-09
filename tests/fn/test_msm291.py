"""Tests for msm291.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from moirais.fn.msm291 import mvsml_convolutional_nn_eq_14_12


def test_msm291_basic():
    """Test basic functionality."""
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    computed = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    training = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(matrix, computed, the, training, of, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm291_edge():
    """Test edge cases."""
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    computed = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    training = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(matrix, computed, the, training, of, model)
    assert isinstance(result, dict)
