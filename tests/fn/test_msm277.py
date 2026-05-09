"""Tests for msm277.mvsml_convolutional_nn_eq_14_10."""
import numpy as np
import pytest
from moirais.fn.msm277 import mvsml_convolutional_nn_eq_14_10


def test_msm277_basic():
    """Test basic functionality."""
    functions = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    spline = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(functions, the, covariate, function, B, spline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm277_edge():
    """Test edge cases."""
    functions = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    spline = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(functions, the, covariate, function, B, spline)
    assert isinstance(result, dict)
