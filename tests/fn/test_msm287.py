"""Tests for msm287.mvsml_convolutional_nn_eq_14_11."""
import numpy as np
import pytest
from moirais.fn.msm287 import mvsml_convolutional_nn_eq_14_11


def test_msm287_basic():
    """Test basic functionality."""
    smoothed = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(smoothed, solution, of, t, can, be)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm287_edge():
    """Test edge cases."""
    smoothed = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(smoothed, solution, of, t, can, be)
    assert isinstance(result, dict)
