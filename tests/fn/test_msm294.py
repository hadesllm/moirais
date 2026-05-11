"""Tests for msm294.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from morie.fn.msm294 import mvsml_convolutional_nn_eq_14_13


def test_msm294_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(where, XE, the, design, matrix, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm294_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(where, XE, the, design, matrix, of)
    assert isinstance(result, dict)
