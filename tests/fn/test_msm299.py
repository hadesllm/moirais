"""Tests for msm299.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from morie.fn.msm299 import mvsml_convolutional_nn_eq_14_14


def test_msm299_basic():
    """Test basic functionality."""
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    This = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    also = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(xT, n, This, model, was, also)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm299_edge():
    """Test edge cases."""
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    This = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    also = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(xT, n, This, model, was, also)
    assert isinstance(result, dict)
