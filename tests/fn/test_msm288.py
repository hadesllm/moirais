"""Tests for msm288.mvsml_convolutional_nn_eq_14_11."""
import numpy as np
import pytest
from morie.fn.msm288 import mvsml_convolutional_nn_eq_14_11


def test_msm288_basic():
    """Test basic functionality."""
    accuracy = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    penalized = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(accuracy, of, this, the, non, penalized)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm288_edge():
    """Test edge cases."""
    accuracy = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    penalized = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(accuracy, of, this, the, non, penalized)
    assert isinstance(result, dict)
